"""FastAPI server for IslandSense dashboard.

Serves:
- Static frontend build
- /api/dashboard endpoint with transformed CSV data
"""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="IslandSense API", version="0.1.0")

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directory
DATA_DIR = Path(__file__).parent.parent.parent / "data"


def get_band_from_risk(risk_score: float) -> str:
    """Convert risk score (0-100) to band.

    0-20: green (Low Risk)
    21-50: amber (Moderate Risk)
    51-100: red (High Risk)
    """
    if risk_score > 50:
        return "red"
    elif risk_score > 20:
        return "amber"
    else:
        return "green"


def scale_risk_to_100(risk_score: float) -> int:
    """Scale weekly risk score to 0-100 scale.

    Weekly risk is counted as sum of daily risks (0-7 scale typically).
    7 = maximum risk (all 7 days are high risk).
    """
    # Scale: 0 = 0, 7 = 100 (max 7 days in a week)
    return int(min(100, (risk_score / 7) * 100))


def format_date_short(date_str: str) -> tuple[str, str]:
    """Convert ISO date to 'TUE', '16 NOV' format."""
    dt = datetime.fromisoformat(date_str)
    day_name = dt.strftime("%a").upper()  # TUE
    date_formatted = dt.strftime("%d %b").upper()  # 16 NOV
    return day_name, date_formatted


def generate_bullets(
    category: str, daily_df: pd.DataFrame, weekly_risk: float
) -> list[str]:
    """Generate summary bullets for a category."""
    bullets = []

    # Count high-risk days
    cat_daily = daily_df[daily_df["category"] == category]
    red_days = cat_daily[cat_daily["band"] == "red"]
    amber_days = cat_daily[cat_daily["band"] == "amber"]

    if len(red_days) > 0:
        day_names = []
        for _, row in red_days.iterrows():
            day_name, _ = format_date_short(row["date"])
            day_names.append(day_name.capitalize())
        bullets.append(f"{len(red_days)} high-risk days ({', '.join(day_names)})")
    elif len(amber_days) > 0:
        day_names = []
        for _, row in amber_days.iterrows():
            day_name, _ = format_date_short(row["date"])
            day_names.append(day_name.capitalize())
        bullets.append(f"{len(amber_days)} moderate-risk days ({', '.join(day_names)})")
    else:
        bullets.append("All days low risk")

    # Add insight about drivers
    if category == "fresh" and weekly_risk > 30:
        bullets.append("Monitor beam-sea sailings for perishables")
    elif category == "fuel" and weekly_risk > 30:
        bullets.append("Fuel demand spikes on high-wind days")
    else:
        bullets.append("Weather conditions favorable")

    return bullets


def load_dashboard_data() -> dict:
    """Load and transform CSV data into frontend JSON structure."""
    # Load CSVs
    daily_risk_df = pd.read_csv(DATA_DIR / "daily_risk.csv")
    pd.read_csv(DATA_DIR / "weekly_risk.csv")  # Validate file exists
    scenario_impact_df = pd.read_csv(DATA_DIR / "scenario_impact.csv")
    sailing_contrib_df = pd.read_csv(DATA_DIR / "sailing_contrib.csv")
    sailing_deltas_df = pd.read_csv(DATA_DIR / "sailing_scenario_deltas.csv")

    # Get last update timestamp (use most recent CSV modification time)
    csv_times = [
        os.path.getmtime(DATA_DIR / "daily_risk.csv"),
        os.path.getmtime(DATA_DIR / "weekly_risk.csv"),
    ]
    last_update = datetime.fromtimestamp(max(csv_times)).strftime("%b %d, %H:%M")

    # Build daily strip
    days = []

    # Calculate weekly scores as sum of daily absolute risk scores, averaged
    # Use risk_baseline column which is already scaled 0-100
    fresh_daily_risks = daily_risk_df[daily_risk_df["category"] == "fresh"][
        "risk_baseline"
    ].tolist()
    fuel_daily_risks = daily_risk_df[daily_risk_df["category"] == "fuel"][
        "risk_baseline"
    ].tolist()

    # Weekly score = average of daily risk scores (already 0-100)
    # This gives a meaningful weekly aggregate
    fresh_score = (
        int(sum(fresh_daily_risks) / len(fresh_daily_risks)) if fresh_daily_risks else 0
    )
    fuel_score = (
        int(sum(fuel_daily_risks) / len(fuel_daily_risks)) if fuel_daily_risks else 0
    )

    for day_idx in sorted(daily_risk_df["day_index"].unique()):
        day_data = daily_risk_df[daily_risk_df["day_index"] == day_idx]
        fresh_day = day_data[day_data["category"] == "fresh"].iloc[0]
        fuel_day = day_data[day_data["category"] == "fuel"].iloc[0]

        day_name, date_formatted = format_date_short(fresh_day["date"])

        # Use absolute risk scores from risk_baseline (already 0-100 scale)
        fresh_pct = int(fresh_day["risk_baseline"])
        fuel_pct = int(fuel_day["risk_baseline"])

        # Band based on worst category
        day_band = (
            fresh_day["band"]
            if fresh_day["band"] == "red"
            or (fresh_day["band"] == "amber" and fuel_day["band"] != "red")
            else fuel_day["band"]
        )

        days.append(
            {
                "name": day_name,
                "date": date_formatted,
                "fresh": fresh_pct,
                "fuel": fuel_pct,
                "band": day_band,
            }
        )

    # Sailings for all days (grouped by day_index)
    sailings_by_day = {}
    for day_idx in sorted(sailing_contrib_df["day_index"].unique()):
        day_sailings = sailing_contrib_df[
            sailing_contrib_df["day_index"] == day_idx
        ].head(10)
        day_list = []
        for _, row in day_sailings.iterrows():
            etd = datetime.fromisoformat(row["etd_iso"])
            # Show risk with 1 decimal place for small values, round for larger
            risk_pct = row["p_sail"] * 100
            if risk_pct < 1:
                risk_display = round(risk_pct, 1)
            else:
                risk_display = round(risk_pct)
            sailing_entry = {
                "time": etd.strftime("%H:%M"),
                "id": row["sailing_id"].split("_")[0]
                + row["sailing_id"][-4:],  # Shorten ID
                "route": row["route"],
                "risk": risk_display,
                "freshExp": round(row["fresh_units"], 1),
                "fuelExp": round(row["fuel_units"], 2),
            }
            # Add feature columns if present
            if "WOTDI" in row:
                sailing_entry["wotdi"] = round(float(row["WOTDI"]), 2)
            if "BSEF" in row:
                sailing_entry["bsef"] = round(float(row["BSEF"]), 2)
            if "gust_max_3h" in row:
                sailing_entry["gust"] = round(float(row["gust_max_3h"]), 1)
            if "tide_gate_margin" in row:
                sailing_entry["tide"] = round(float(row["tide_gate_margin"]), 0)
            day_list.append(sailing_entry)
        sailings_by_day[int(day_idx)] = day_list  # Convert numpy.int64 to Python int

    # Keep backwards compatibility - sailings is day 0
    sailings = sailings_by_day.get(0, [])

    # Scenario A impact
    scenario_a_fresh = scenario_impact_df[
        (scenario_impact_df["scenario_id"] == "scenario_A")
        & (scenario_impact_df["category"] == "fresh")
    ].iloc[0]
    scenario_a_fuel = scenario_impact_df[
        (scenario_impact_df["scenario_id"] == "scenario_A")
        & (scenario_impact_df["category"] == "fuel")
    ].iloc[0]

    # Scenario B impact
    scenario_b_fresh = scenario_impact_df[
        (scenario_impact_df["scenario_id"] == "scenario_B")
        & (scenario_impact_df["category"] == "fresh")
    ].iloc[0]
    scenario_b_fuel = scenario_impact_df[
        (scenario_impact_df["scenario_id"] == "scenario_B")
        & (scenario_impact_df["category"] == "fuel")
    ].iloc[0]

    # Helper to calculate scenario scores using proportional reduction
    def calc_scenario_score(
        baseline_score: int, csv_baseline: float, csv_scenario: float
    ) -> int:
        """Apply proportional reduction from CSV scenario to our calculated baseline."""
        if csv_baseline == 0:
            return baseline_score
        # Apply same proportional change
        ratio = csv_scenario / csv_baseline
        return int(baseline_score * ratio)

    # Build scenario sailing actions
    def build_scenario_sailings(scenario_id: str) -> list[dict]:
        scenario_deltas = (
            sailing_deltas_df[sailing_deltas_df["scenario_id"] == scenario_id]
            .sort_values("delta_fresh", ascending=False)
            .head(10)
        )  # Sort by highest impact first

        result = []
        for _, row in scenario_deltas.iterrows():
            etd = datetime.fromisoformat(row["etd_iso"])
            day_name, date_formatted = format_date_short(row["date"])

            # Use suggested_action from CSV if available, otherwise generate
            if "suggested_action" in row and pd.notna(row["suggested_action"]):
                action = row["suggested_action"]
            else:
                # Fallback to old logic
                fresh_pct = (
                    round(row["delta_fresh"] / row["contrib_fresh"] * 100)
                    if row["contrib_fresh"] > 0
                    else 0
                )
                fuel_pct = (
                    round(row["delta_fuel"] / row["contrib_fuel"] * 100)
                    if row["contrib_fuel"] > 0
                    else 0
                )

                if fuel_pct > 0 and fresh_pct > 0:
                    action = f"Bring forward {fresh_pct}% Fresh, {fuel_pct}% Fuel"
                elif fresh_pct > 0:
                    action = f"Bring forward {fresh_pct}% Fresh"
                else:
                    action = "No action needed"

            result.append(
                {
                    "date": date_formatted,
                    "time": etd.strftime("%H:%M"),
                    "id": row["sailing_id"].split("_")[0] + row["sailing_id"][-4:],
                    "route": row["route"],
                    "freshDelta": round(
                        row["delta_fresh"], 2
                    ),  # Positive = risk reduction
                    "fuelDelta": round(row["delta_fuel"], 3),
                    "action": action,
                }
            )

        return result

    # Recommended plan (use scenario A as recommended)
    # Use our calculated weekly scores as baseline, apply proportional reduction from CSV
    fresh_scenario_a = calc_scenario_score(
        fresh_score,
        scenario_a_fresh["weekly_risk_baseline"],
        scenario_a_fresh["weekly_risk_scenario"],
    )
    fuel_scenario_a = calc_scenario_score(
        fuel_score,
        scenario_a_fuel["weekly_risk_baseline"],
        scenario_a_fuel["weekly_risk_scenario"],
    )

    # Get policy name from CSV (now comes from optimizer)
    policy_name = scenario_a_fresh.get("scenario_name", "Optimized Policy")
    forward_frac_a = scenario_a_fresh.get("forward_frac", 0.0)

    # Calculate trailers avoided based on risk score delta (more meaningful for demo)
    # If CSV trailers_avoided is 0, use risk score delta as proxy (1 risk point = 0.1 trailers)
    trailers_avoided_a = scenario_a_fuel["trailers_avoided"]
    if trailers_avoided_a == 0 and fuel_score > fuel_scenario_a:
        trailers_avoided_a = round((fuel_score - fuel_scenario_a) * 0.1, 1)

    recommended_plan = {
        "policy": policy_name,
        "forwardFrac": forward_frac_a,
        "fresh": {
            "baseline": fresh_score,
            "scenario": fresh_scenario_a,
            "delta": fresh_scenario_a - fresh_score,
            "hoursAvoided": round(scenario_a_fresh["hours_avoided"], 1),
        },
        "fuel": {
            "baseline": fuel_score,
            "scenario": fuel_scenario_a,
            "delta": fuel_scenario_a - fuel_score,
            "trailersAvoided": trailers_avoided_a,
        },
    }

    return {
        "lastUpdate": last_update,
        "fresh": {
            "score": fresh_score,
            "band": get_band_from_risk(fresh_score),
            "bullets": generate_bullets("fresh", daily_risk_df, fresh_score),
        },
        "fuel": {
            "score": fuel_score,
            "band": get_band_from_risk(fuel_score),
            "bullets": generate_bullets("fuel", daily_risk_df, fuel_score),
        },
        "recommendedPlan": recommended_plan,
        "days": days,
        "sailings": sailings,
        "sailingsByDay": sailings_by_day,
        "scenarios": {
            "A": {
                "name": "Optimized",
                "description": scenario_a_fresh.get(
                    "scenario_name", "Optimized: forward 10%"
                ),
                "forwardFrac": scenario_a_fresh.get("forward_frac", 0.0),
                "fresh": {
                    "baseline": fresh_score,
                    "scenario": fresh_scenario_a,
                    "delta": fresh_scenario_a - fresh_score,
                    "hoursAvoided": round(scenario_a_fresh["hours_avoided"], 1),
                },
                "fuel": {
                    "baseline": fuel_score,
                    "scenario": fuel_scenario_a,
                    "delta": fuel_scenario_a - fuel_score,
                    "trailersAvoided": round(scenario_a_fuel["trailers_avoided"], 0),
                },
                "sailings": build_scenario_sailings("scenario_A"),
            },
            "B": {
                "name": "Aggressive",
                "description": scenario_b_fresh.get(
                    "scenario_name", "Bring forward 10% + air-lift 5%"
                ),
                "forwardFrac": scenario_b_fresh.get("forward_frac", 0.0),
                "fresh": {
                    "baseline": fresh_score,
                    "scenario": calc_scenario_score(
                        fresh_score,
                        scenario_b_fresh["weekly_risk_baseline"],
                        scenario_b_fresh["weekly_risk_scenario"],
                    ),
                    "delta": calc_scenario_score(
                        fresh_score,
                        scenario_b_fresh["weekly_risk_baseline"],
                        scenario_b_fresh["weekly_risk_scenario"],
                    )
                    - fresh_score,
                    "hoursAvoided": round(scenario_b_fresh["hours_avoided"], 1),
                },
                "fuel": {
                    "baseline": fuel_score,
                    "scenario": calc_scenario_score(
                        fuel_score,
                        scenario_b_fuel["weekly_risk_baseline"],
                        scenario_b_fuel["weekly_risk_scenario"],
                    ),
                    "delta": calc_scenario_score(
                        fuel_score,
                        scenario_b_fuel["weekly_risk_baseline"],
                        scenario_b_fuel["weekly_risk_scenario"],
                    )
                    - fuel_score,
                    "trailersAvoided": round(
                        (
                            fuel_score
                            - calc_scenario_score(
                                fuel_score,
                                scenario_b_fuel["weekly_risk_baseline"],
                                scenario_b_fuel["weekly_risk_scenario"],
                            )
                        )
                        * 0.1,
                        1,
                    )
                    if scenario_b_fuel["trailers_avoided"] == 0
                    else round(scenario_b_fuel["trailers_avoided"], 0),
                },
                "sailings": build_scenario_sailings("scenario_B"),
            },
        },
    }


@app.get("/api/dashboard")
def get_dashboard():
    """Get dashboard data for frontend."""
    return load_dashboard_data()


# Mount static frontend (production build)
frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/")
    def serve_frontend():
        """Serve frontend index.html."""
        return FileResponse(frontend_dist / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
