# DATASETS.md — IslandSense MVP

> **Purpose:** Single place that answers
> “Which CSV is which, who writes it, and who reads it?”

IslandSense MVP has three main stages:

1. **Synthesis / raw inputs** → synthetic world (routes, sailings, metocean, tides, exposure)
2. **Training / prediction** → per-sailing features, model, `p_sail`
3. **Aggregation / UI** → daily & weekly JDI, scenarios, drilldown tables

Note: all **economic / prescriptive parameters** (costs of actions, penalties,
budgets, shift limits, etc.) live in `config_mvp.yaml` and are **not**
represented as CSV datasets in this MVP. The CSVs here stop at exposure
and probabilities; the "what to do" layer is driven by config + code only.

Concretely, the prescriptive / optimisation layer reads from
`config_mvp.yaml` keys such as:

- `costs.*` (penalty and action cost parameters),
- `shift_limits.*` (max forward / air fractions),
- `budgets.*` (optional weekly caps),
- `scenarios.*` (static or optimiser-derived `alpha` values per category).

It then operates entirely on:

- `per_sailing_predictions.csv`  → per-sailing `p_sail` and exposure
- `daily_jdi.csv` / `weekly_jdi.csv` → aggregated `E_loss` / JDI

to compute net benefit and choose a recommended weekly plan.

No additional CSVs are required to support the "tiny optimiser" version; the
schemas in this document remain valid. If we ever promote to a richer
prescriptive product, we can introduce dedicated cost / contract tables in
a future revision.

This document lists each dataset, its schema, **producer**, and **consumers**.

---

## 1. Synthesis & Raw Inputs

These are created by `synth.py` (or hand-written) and treated as **source of truth** for the MVP.

### 1.1 `routes.csv`

**Role:** Static route definitions.

**Producer:**

- `src/synth.py`

**Consumers:**

- `src/prep.py` (joins onto sailings to build features)
- `src/model.py` (indirectly via `per_sailing_training.csv`)

**Columns (canonical):**

- `route_id` (str)
- `origin_port` (str)
- `dest_port` (str)
- `distance_nm` (float)
- `approach_heading_deg` (float) — average approach bearing into Jersey

---

### 1.2 `vessels.csv`

**Role:** Static vessel definitions.

**Producer:**

- `src/synth.py`

**Consumers:**

- `src/prep.py`

**Columns:**

- `vessel_id` (str)
- `vessel_class` (str) — e.g. “RoRo”, “fast_ferry”
- `route_id` (str) — or link via a mapping table if routes are shared
- `max_wave_operability_m` (float)
- `service_speed_kt` (float)

---

### 1.3 `sailings.csv`

**Role:** Scheduled sailings (one row per vessel event).

**Producer:**

- `src/synth.py`

**Consumers:**

- `src/prep.py`
- `src/aggregate.py` (via derived tables)

**Columns:**

- `sailing_id` (str)
- `route_id` (str)
- `vessel_id` (str)
- `etd_iso` (str, ISO 8601)
- `eta_iso_sched` (str, ISO 8601)
- *(optional v2)* `bin_id_6h` (int) — not required for the 7-day MVP

---

### 1.4 `metocean.csv`

**Role:** Hourly metocean time series per route.

**Producer:**

- `src/synth.py`

**Consumers:**

- `src/prep.py` (to compute physics features)

**Columns:**

- `ts_iso` (str, ISO 8601)
- `route_id` (str)
- `wind_kts` (float)
- `wind_dir_deg` (float)
- `gust_kts` (float)
- `hs_m` (float)
- `tp_s` (float)
- `wave_dir_deg` (float)

---

### 1.5 `tides.csv`

**Role:** Hourly tide time series per port/route.

**Producer:**

- `src/synth.py`

**Consumers:**

- `src/prep.py` (tide_gate_margin / related features)

**Columns:**

- `ts_iso` (str, ISO 8601)
- `port_id` (str)
- `tide_height_m` (float)
- *(optional v2)* `tide_stream_dir_deg` (float)
- *(optional v2)* `tide_stream_kt` (float)

---

### 1.6 `exposure_by_sailing.csv`

**Role:** Synthetic category exposure per sailing.

**Producer:**

- `src/synth.py`

**Consumers:**

- `src/prep.py` (for joining exposure)
- `src/aggregate.py` (implicitly, via `per_sailing_predictions.csv`)

**Columns:**

- `sailing_id` (str)
- `fresh_units` (float) — e.g. pallets of fresh food
- `fuel_units` (float) — e.g. trailer-equivalents for fuel

> **Note:** This is explicitly **synthetic** in the MVP. Real freight/customs data would replace this file later.

---

## 2. Training & Prediction

These files are produced by `prep.py` + `model.py`. They are **intermediate engineering artefacts**, not directly used by the UI.

### 2.1 `per_sailing_training.csv`

**Role:** Final training table for the per-sailing classifier.

**Producer:**

- `src/prep.py`

**Consumers:**

- `src/model.py` (training)

**Columns (MVP core):**

- Identifiers:
  - `sailing_id`
  - `route_id`
  - `vessel_id`
  - `vessel_class`
- Temporal:
  - `etd_iso`
  - `day_of_week` (int, 0–6)
  - `month` (int, 1–12)
- Physics / environment:
  - `Hs_eff` (or `hs_m` as proxy)
  - `WOTDI`
  - `BSEF`
  - `gust_max_3h`
  - `tide_gate_margin`
  - `prior_24h_delay_route`
- Labels:
  - `delay_min`
  - `cancelled` (0/1)
  - `disruption` (0/1) — **main classification label**

*(Additional features allowed as long as they are defined in SPEC.)*

---

### 2.2 `model.pkl` (or equivalent)

**Role:** Serialized per-sailing classifier.

**Producer:**

- `src/model.py` (training runner)

**Consumers:**

- `src/model.py` (prediction)
- `src/app.py` (optional if you do live inference; not required if you precompute predictions)

---

### 2.3 `per_sailing_predictions.csv`

**Role:** Input to M3 aggregation; per-sailing probabilities + exposure.

**Producer:**

- `src/model.py` (prediction runner)

**Consumers:**

- `src/aggregate.py`

**Columns (required):**

- `sailing_id` (str)
- `route` (str) — for display
- `vessel` (str) — for display
- `etd_iso` (str, ISO 8601)
- `p_sail` (float) — model probability of disruption
- `fresh_units` (float)
- `fuel_units` (float)

> This is the **single** input file for M3 aggregation in the MVP.

---

## 3. Aggregation & UI Outputs

These are produced by `aggregate.py` and read by the Streamlit app (`app.py`). They are the only datasets the UI needs.

### 3.1 `sailing_contrib.csv`

**Role:** Per-sailing drilldown for each forecast day.

**Producer:**

- `src/aggregate.py`

**Consumers:**

- `src/app.py` (day-level drilldown table)

**Columns:**

- `sailing_id` (str)
- `day_index` (int, 0–6)
- `date` (str, `YYYY-MM-DD`)
- `route` (str)
- `vessel` (str)
- `etd_iso` (str, ISO 8601)
- `p_sail` (float)
- `fresh_units` (float)
- `fuel_units` (float)
- `contrib_fresh` (float) — `p_sail * fresh_units`
- `contrib_fuel` (float) — `p_sail * fuel_units`

---

### 3.2 `daily_jdi.csv`

**Role:** Daily JDI per category for D0..D6 (7-day strip).

**Producer:**

- `src/aggregate.py`

**Consumers:**

- `src/app.py` (7-day JDI tiles)

**Columns:**

- `day_index` (int, 0–6)
- `date` (str, `YYYY-MM-DD`)
- `category` (str: `"fresh"` or `"fuel"`)
- `E_loss` (float) — sum of `p_sail * exposure` for the day
- `JDI_baseline` (int, 0–100)
- `band` (str: `"green" | "amber" | "red"`)

---

### 3.3 `weekly_jdi.csv`

**Role:** Weekly JDI baseline + per-scenario JDI, per category.

**Producer:**

- `src/aggregate.py`

**Consumers:**

- `src/app.py` (weekly header card and scenario cards)

**Columns (MVP shape):**

- `category` (str: `"fresh"` or `"fuel"`)
- `weekly_JDI_baseline` (int, 0–100)
- `weekly_JDI_scenario_A` (int, 0–100)
- `weekly_JDI_scenario_B` (int, 0–100)

> Scenario IDs must match `config.yaml.scenarios[*].id`.

---

### 3.4 `scenario_impact.csv`

**Role:** Weekly impact per scenario/category (for copy on scenario cards).

**Producer:**

- `src/aggregate.py`

**Consumers:**

- `src/app.py` (scenario card text: “~X hours avoided”, “~Y trailers avoided”)

**Columns:**

- `scenario_id` (str: `"scenario_A"`, `"scenario_B"`, ...)
- `category` (str: `"fresh"` or `"fuel"`)
- `weekly_JDI_baseline` (int)
- `weekly_JDI_scenario` (int)
- `delta_JDI` (int) — baseline - scenario
- `hours_avoided` (float, 1 decimal) — meaningful for Fresh
- `trailers_avoided` (float, 1 decimal) — meaningful for Fuel

---

## 4. Optional / Nice-to-Have Outputs

These are optional; only build them if you have time.

### 4.1 `ops_brief_week.csv` (or PDF)

**Role:** Exported “Ops Brief” for sharing (WhatsApp/Teams/email).

**Producer:**

- `src/app.py` (via Streamlit download)

**Consumers:**

- Humans (ops / judges)

**Suggested content:**

- Header:
  - generation timestamp
  - horizon: “Next 7 days”
- Summary rows:
  - For Fresh: weekly JDI baseline, recommended scenario, hours avoided
  - For Fuel: weekly JDI baseline, recommended scenario, trailers avoided
- Top sailings:
  - 5–10 highest-contribution sailings from `sailing_contrib.csv`
  - `route, vessel, etd_iso, p_sail, contrib_fresh, contrib_fuel`

---

## 5. Script → Dataset Map (at a glance)

**`src/synth.py`**

- Writes:
  - `routes.csv`
  - `vessels.csv`
  - `sailings.csv`
  - `metocean.csv`
  - `tides.csv`
  - `exposure_by_sailing.csv`

**`src/prep.py`**

- Reads:
  - all of the above
- Writes:
  - `per_sailing_training.csv`

**`src/model.py`**

- Reads:
  - `per_sailing_training.csv`
- Writes:
  - `model.pkl`
  - `per_sailing_predictions.csv`

**`src/aggregate.py`**

- Reads:
  - `per_sailing_predictions.csv`
- Writes:
  - `sailing_contrib.csv`
  - `daily_jdi.csv`
  - `weekly_jdi.csv`
  - `scenario_impact.csv`

**`src/app.py` (Streamlit)**

- Reads:
  - `sailing_contrib.csv`
  - `daily_jdi.csv`
  - `weekly_jdi.csv`
  - `scenario_impact.csv`
- Optionally writes:
  - `ops_brief_week.csv` (via download)

---

If you keep this file up to date, it becomes your “don’t think about file names” oracle during the sprint. If something new shows up (e.g. a calibration JSON, or a storm-replay CSV), just add a small row under section 3 or 4 instead of letting it float around in your head.
