"""Schema definitions and validators for IslandSense CSV data."""

from dataclasses import dataclass
from typing import List
import pandas as pd


# Column name constants for each CSV
class SailingColumns:
    """Column names for sailings.csv"""

    SAILING_ID = "sailing_id"
    ROUTE = "route"
    VESSEL = "vessel"
    ETD_ISO = "etd_iso"
    ETA_ISO = "eta_iso"
    HEAD_DEG = "head_deg"

    @classmethod
    def all(cls) -> List[str]:
        return [
            cls.SAILING_ID,
            cls.ROUTE,
            cls.VESSEL,
            cls.ETD_ISO,
            cls.ETA_ISO,
            cls.HEAD_DEG,
        ]


class StatusColumns:
    """Column names for status.csv"""

    SAILING_ID = "sailing_id"
    STATUS = "status"
    DELAY_MIN = "delay_min"

    @classmethod
    def all(cls) -> List[str]:
        return [cls.SAILING_ID, cls.STATUS, cls.DELAY_MIN]


class MetoceanColumns:
    """Column names for metocean.csv"""

    TS_ISO = "ts_iso"
    WIND_KTS = "wind_kts"
    WIND_DIR_DEG = "wind_dir_deg"
    GUST_KTS = "gust_kts"
    HS_M = "hs_m"
    TP_S = "tp_s"
    WAVE_DIR_DEG = "wave_dir_deg"

    @classmethod
    def all(cls) -> List[str]:
        return [
            cls.TS_ISO,
            cls.WIND_KTS,
            cls.WIND_DIR_DEG,
            cls.GUST_KTS,
            cls.HS_M,
            cls.TP_S,
            cls.WAVE_DIR_DEG,
        ]


class TideColumns:
    """Column names for tides.csv"""

    TS_ISO = "ts_iso"
    TIDE_M = "tide_m"

    @classmethod
    def all(cls) -> List[str]:
        return [cls.TS_ISO, cls.TIDE_M]


class ExposureColumns:
    """Column names for exposure_by_sailing.csv"""

    SAILING_ID = "sailing_id"
    FRESH_UNITS = "fresh_units"
    FUEL_UNITS = "fuel_units"

    @classmethod
    def all(cls) -> List[str]:
        return [cls.SAILING_ID, cls.FRESH_UNITS, cls.FUEL_UNITS]


class MyShipmentColumns:
    """Column names for my_sailings.csv (optional)"""

    SHIPMENT_ID = "shipment_id"
    SAILING_ID = "sailing_id"
    COMMODITY = "commodity"
    QTY_UNITS = "qty_units"

    @classmethod
    def all(cls) -> List[str]:
        return [cls.SHIPMENT_ID, cls.SAILING_ID, cls.COMMODITY, cls.QTY_UNITS]


# Dataclass representations (lightweight, no Pydantic per cut rule)
@dataclass
class Sailing:
    """Represents a single sailing."""

    sailing_id: str
    route: str
    vessel: str
    etd_iso: str
    eta_iso: str
    head_deg: float


@dataclass
class Status:
    """Represents the status/outcome of a sailing."""

    sailing_id: str
    status: str  # "arrived" or "cancelled"
    delay_min: int


@dataclass
class Metocean:
    """Represents metocean conditions at a point in time."""

    ts_iso: str
    wind_kts: float
    wind_dir_deg: float
    gust_kts: float
    hs_m: float
    tp_s: float
    wave_dir_deg: float


@dataclass
class Tide:
    """Represents tide height at a point in time."""

    ts_iso: str
    tide_m: float


@dataclass
class ExposureBySailing:
    """Represents Fresh/Fuel exposure per sailing."""

    sailing_id: str
    fresh_units: float
    fuel_units: float


# Validation helpers
def assert_required_columns(
    df: pd.DataFrame, expected: List[str], name: str = "DataFrame"
):
    """Assert that DataFrame contains all expected columns."""
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")

    extra = set(df.columns) - set(expected)
    if extra:
        print(f"Warning: {name} has extra columns: {extra}")


def validate_sailings(df: pd.DataFrame) -> None:
    """Validate sailings.csv schema."""
    assert_required_columns(df, SailingColumns.all(), "sailings.csv")
    assert df[SailingColumns.SAILING_ID].is_unique, "sailing_id must be unique"
    assert df[SailingColumns.HEAD_DEG].between(0, 360).all(), (
        "head_deg must be in [0, 360]"
    )


def validate_status(df: pd.DataFrame) -> None:
    """Validate status.csv schema."""
    assert_required_columns(df, StatusColumns.all(), "status.csv")
    valid_statuses = {"arrived", "cancelled"}
    assert df[StatusColumns.STATUS].isin(valid_statuses).all(), (
        f"status must be one of {valid_statuses}"
    )
    assert (df[StatusColumns.DELAY_MIN] >= 0).all(), "delay_min must be >= 0"


def validate_metocean(df: pd.DataFrame) -> None:
    """Validate metocean.csv schema."""
    assert_required_columns(df, MetoceanColumns.all(), "metocean.csv")
    assert (df[MetoceanColumns.WIND_KTS] >= 0).all(), "wind_kts must be >= 0"
    assert (df[MetoceanColumns.GUST_KTS] >= 0).all(), "gust_kts must be >= 0"
    assert (df[MetoceanColumns.HS_M] >= 0).all(), "hs_m must be >= 0"
    assert df[MetoceanColumns.WIND_DIR_DEG].between(0, 360).all(), (
        "wind_dir_deg must be in [0, 360]"
    )
    assert df[MetoceanColumns.WAVE_DIR_DEG].between(0, 360).all(), (
        "wave_dir_deg must be in [0, 360]"
    )


def validate_tides(df: pd.DataFrame) -> None:
    """Validate tides.csv schema."""
    assert_required_columns(df, TideColumns.all(), "tides.csv")


def validate_exposure(df: pd.DataFrame) -> None:
    """Validate exposure_by_sailing.csv schema."""
    assert_required_columns(df, ExposureColumns.all(), "exposure_by_sailing.csv")
    assert (df[ExposureColumns.FRESH_UNITS] >= 0).all(), "fresh_units must be >= 0"
    assert (df[ExposureColumns.FUEL_UNITS] >= 0).all(), "fuel_units must be >= 0"
