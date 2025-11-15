Here’s a SCHEMA.md you can drop into the repo and then write a** **`synth_data.py` against.

---

```markdown
# SCHEMA.md — Synthetic Data for IslandSense MVP

**Scope:**  
This document defines the **schemas and synthesis rules** for the synthetic datasets used by the IslandSense MVP.

All data described here is **fully synthetic**, contains **no personal information**, and is designed only to:

- Train a **per-sailing disruption model**;
- Provide a **7-day demo horizon** with interesting cases (some disruptions, some near-misses);
- Drive the **"Fresh / Fuel" 7-day JDI strip, weekly JDI, and per-sailing drilldown**.

The schemas are intentionally small and opinionated to keep data generation doable in a few hours.

---

## 0. Overall shape

We generate:

- A **history period** (e.g. 90–365 days) → training + validation.
- A **demo horizon** (e.g. the next 7 days from “now”) → inference + UI.

Key relations:

- `sailings.csv` is the **spine** (one row per sailing).
- `status.csv` attaches **labels** to each sailing.
- `metocean.csv` and `tides.csv` are **time series** used to compute physics features.
- `exposure_by_sailing.csv` attaches **category exposure** (Fresh/Fuel) to each sailing.

You should be able to regenerate everything from a fixed `rng_seed`.

---

## 1. `sailings.csv`

**Purpose:** Canonical list of scheduled sailings (history + future).  
**Grain:** 1 row = 1 sailing.

### 1.1 Columns

| Column      | Type    | Description                                                |
| ----------- | ------- | ---------------------------------------------------------- |
| `sailing_id`| string  | Unique ID per sailing (`route_code + date + seq`).        |
| `route`     | string  | Route name, e.g. `"Portsmouth→Jersey"`.                   |
| `vessel`    | string  | Vessel name, e.g. `"Island Spirit"`.                      |
| `etd_iso`   | string  | Scheduled departure (UTC, ISO 8601).                      |
| `eta_iso`   | string  | Scheduled arrival (UTC, ISO 8601).                        |
| `head_deg`  | float   | Approx. heading on approach to Jersey (0–360 degrees).    |

### 1.2 Synthesis rules

1. **Routes (small, fixed set):**

   Example:

   - `Portsmouth→Jersey`
   - `St. Malo→Jersey`
   - `Poole→Jersey`

2. **Vessels (per route):**

   - Each route has 1–3 vessels (e.g. `"Island Spirit"`, `"Channel Star"`).
   - Assign `head_deg` per route with small noise:
     - `Portsmouth→Jersey`: ~ 220° ± 5°
     - `St. Malo→Jersey`: ~ 300° ± 5°
     - `Poole→Jersey`: ~ 200° ± 5°

3. **Schedule:**

   - History: generate e.g. 6 months of sailings.
   - For each route, generate departures:
     - 1–3 sailings per day at fixed local times (e.g. 06:00, 12:00, 18:00).
   - Convert to UTC and ISO strings.
   - Set `eta_iso = etd_iso + nominal_duration` (e.g. 8–10h depending on route).

4. **Demo horizon:**

   - Ensure at least 3–5 sailings **inside the “next 72h”** window.
   - For those, you’ll later tilt the metocean data to create a storm (for interesting risk).

---

## 2. `metocean.csv`

**Purpose:** Marine weather on a regular grid used to compute physics features.  
**Grain:** 1 row = 1 timestamp.

### 2.1 Columns

| Column         | Type   | Description                                      |
| -------------- | ------ | ------------------------------------------------ |
| `ts_iso`       | string | Timestamp (UTC, ISO 8601).                      |
| `wind_kts`     | float  | Mean wind speed (knots).                        |
| `wind_dir_deg` | float  | Mean wind direction (degrees, 0–360).          |
| `gust_kts`     | float  | Max gust (knots).                               |
| `hs_m`         | float  | Significant wave height (m).                    |
| `tp_s`         | float  | Peak period (seconds).                          |
| `wave_dir_deg` | float  | Mean wave direction (degrees, 0–360).          |

### 2.2 Synthesis rules

1. **Time grid:**

   - Use hourly grid across **history + demo** period.
   - Example: from `today - 180 days` to `today + 7 days`, every hour.

2. **Base climate (calm-ish):**

   - `wind_kts`: ~ Normal(15, 5), clipped to [0, 50].
   - `gust_kts`: `wind_kts + Normal(5, 3)`, clipped to [0, 60].
   - `hs_m`: ~ Normal(1.0, 0.5), clipped to [0, 5].
   - `tp_s`: ~ Normal(6, 2), clipped to [3, 14].
   - `wind_dir_deg`: uniform on [0, 360).
   - `wave_dir_deg`: correlated with wind (e.g. wind_dir + Normal(0, 30)).

3. **Inject a few “storm periods”:**

   - Choose 3–6 random **storm windows** (8–24 hours each).
   - During storms:
     - `wind_kts`: bump to ~ Normal(35, 5).
     - `gust_kts`: ~ wind + Normal(10, 5).
     - `hs_m`: ~ Normal(3.0, 0.7).
     - Optionally align `wave_dir_deg` to make beam-sea conditions high for one route.

4. **Demo storm (for the 72h horizon):**

   - Ensure one storm overlaps some sailings in the “next 72h” window:
     - This is your “Saturday gale” for the demo.
   - Mark it in your generator (e.g. via comments or scenario ID if you want).

---

## 3. `tides.csv`

**Purpose:** Tide height at or near Jersey used as a simple feature.  
**Grain:** 1 row = 1 timestamp (match metocean timestamps).

### 3.1 Columns

| Column   | Type   | Description                          |
| -------- | ------ | ------------------------------------ |
| `ts_iso` | string | Timestamp (UTC, ISO 8601).          |
| `tide_m` | float  | Tide height (metres).               |

### 3.2 Synthesis rules

1. **Same hourly grid** as `metocean.csv` (same `ts_iso` values).
2. **Tide curve**: simple sinusoid with noise:

   - Choose amplitude (e.g. 2 m), mean level (e.g. 3 m).
   - For timestamp `t` (in hours since start):

     ```text
     tide_m = mean + amplitude * sin( 2π * t / tide_period_hours ) + ε
     ```

   - `tide_period_hours` ~ 12.4 (semi-diurnal approximation).
   - Add small noise ε ~ Normal(0, 0.1).

---

## 4. `status.csv`

**Purpose:** Outcomes per sailing to define the **label**.  
**Grain:** 1 row = 1 sailing.

### 4.1 Columns

| Column      | Type   | Description                                         |
| ----------- | ------ | --------------------------------------------------- |
| `sailing_id`| string | Foreign key to `sailings.csv`.                      |
| `status`    | string | `"arrived"` or `"cancelled"`.                       |
| `delay_min` | int    | Arrival delay in minutes (0 if on time).           |

### 4.2 Synthesis rules

We want **physics → risk → label** to feel plausible:

1. For each sailing:

   - Look up metocean + tide at `etd_iso` (nearest hour).
   - Compute **synthetic risk score** `r_s` using a simple formula like:

     ```text
     rel_wave_deg = wrap(wave_dir_deg - head_deg to [-180, 180])
     bsef = abs(sin(rel_wave_deg_rad)) * hs_m
     high_wind = max(0, wind_kts - wind_threshold)
     high_gust = max(0, gust_kts - gust_threshold)

     r_s = a1 * bsef + a2 * high_wind + a3 * high_gust
     ```

   - Choose small positive coefficients `a1, a2, a3` so:
     - Calm conditions → `r_s` near 0.
     - Stormy beam-sea conditions → `r_s` larger (e.g. 0.5–1.5).

2. Convert `r_s` into disruption probability `p_disrupt_s`:

   ```text
   p_disrupt_s = sigmoid(α * (r_s - β))
```

* Choose α, β so that:
  * Most normal sailings have p ≈ 1–5%.
  * Storm-overlapped sailings have p ≈ 30–60%.

3. Sample label:
   * Draw** **`u ~ Uniform(0,1)`.
   * If** **`u < p_disrupt_s` →** ** **disruption = 1** .
   * Else** ** **disruption = 0** .
4. Derive** **`status` and** **`delay_min`:
   * If** **`disruption = 1`:
     * With some probability (e.g. 30–50%), set** **`status = "cancelled"`,** **`delay_min = 0`.
     * Else** **`status = "arrived"`,** **`delay_min` ~ Uniform(130, 300) (minutes).
   * If** **`disruption = 0`:
     * `status = "arrived"`.
     * `delay_min` ~ mixture of:
       * 70–80%: small noise, e.g. Uniform(0, 15).
       * 20–30%: moderate delays, e.g. Uniform(15, 90).

This gives you a** ****ground-truth label** consistent with weather & waves, which your ML model will try to re-learn from the features.

---

## 5.** **`exposure_by_sailing.csv`

**Purpose:** Approximate how much** ****Fresh** and** ****Fuel** exposure is on each sailing.
**Grain:** 1 row = 1 sailing.

### 5.1 Columns

| Column          | Type   | Description                                   |
| --------------- | ------ | --------------------------------------------- |
| `sailing_id`  | string | FK to `sailings.csv`.                       |
| `fresh_units` | float  | Exposure for Fresh (e.g. pallet equivalents). |
| `fuel_units`  | float  | Exposure for Fuel (e.g. trailer equivalents). |

### 5.2 Synthesis rules

You just need something plausible, not correct:

1. For each route:
   * Assign a** ****mean Fresh load** and** ** **mean Fuel load** :
     * e.g. Portsmouth→Jersey: Fresh ~ 40 pallets, Fuel ~ 1 trailer.
     * St. Malo→Jersey: Fresh ~ 25, Fuel ~ 0.5.
     * Poole→Jersey: Fresh ~ 15, Fuel ~ 0.2.
2. For each sailing:
   * Draw Fresh & Fuel around those means:
     ```text
     fresh_units = max(0, Normal(mean_fresh, mean_fresh * 0.3))
     fuel_units  = max(0, Normal(mean_fuel,  mean_fuel  * 0.3))
     ```
3. To make the** ****demo horizon** visible:
   * Slightly increase loading on some storm-overlapped sailings so impact is visible when shifted (e.g. +30%).

---

## 6. (Optional)** **`my_sailings.csv`

**Purpose:** If you show “My shipments” or similar drill-down.
**Grain:** 1 row = 1 shipment (subset of exposure, per sailing).

For MVP you can skip this or derive it trivially from** **`exposure_by_sailing`.

### 6.1 Minimal schema

| Column          | Type   | Description                    |
| --------------- | ------ | ------------------------------ |
| `shipment_id` | string | Unique ID per shipment.        |
| `sailing_id`  | string | FK to `sailings.csv`.        |
| `commodity`   | string | e.g. "Fresh", "Fuel", "Mixed". |
| `qty_units`   | float  | Units (e.g. pallets).          |

You can generate 1–3 shipments per sailing by splitting** **`fresh_units` or** **`fuel_units` randomly.

---

## 7. Synthetic "Now" & 7-day horizon

The **MVP app** assumes a "now" time and a **7-day forecast horizon**.

* Define in your generator:
  ```text
  DEMO_NOW = some recent timestamp near the end of the history period
  DEMO_HORIZON_DAYS = 7
  ```
* Ensure:
  * There are enough sailings in `[DEMO_NOW, DEMO_NOW + 7 days)`.
  * At least one **storm** overlaps some of those days so that:
    * some days have clearly higher disruption risk,
    * a weekly plan (bring-forward / air) makes a visible difference.
  * Those overlapping sailings have **non-trivial Fresh/Fuel units**.

The same synthetic corpus then serves:

* Training (past 90–180 days).
* Demo (next 7 days from `DEMO_NOW`).

---

## 8. Summary of dependencies

**Generation order (recommended):**

1. **Choose config:**
   * Routes, vessels, storm windows, DEMO_NOW, etc.
2. **Generate time grid:**
   * `metocean.csv` (hourly).
   * `tides.csv` (hourly).
3. **Generate sailings:**
   * `sailings.csv` for history + demo horizon.
4. **Generate exposures:**
   * `exposure_by_sailing.csv`.
5. **Generate status/labels:**
   * Use sailings + metocean + tides + simple physics/threshold rules →** **`status.csv`.
6. (Optional)** ****Generate my_sailings.csv** by splitting exposure.

Once these files exist,** **`prep.py` can:

* Recompute features from metocean/tides,
* Train the classifier from scratch,
* Evaluate, and
* Save** **`model.pkl`.

---

## 9. Sanity checks

After synthesis, run quick checks:

* **Counts:**
  * at least a few thousand historical sailings,
  * at least 5–10 sailings in the next 72h horizon.
* **Label rate:**
  * global disruption rate ≈ 2–8% (tune** **`α, β` as needed).
* **Physics correlation:**
  * average** **`p_disrupt_s` is** ***higher* in storm windows than calm windows.
* **Exposure:**
  * Fresh/Fuel distributions non-zero and roughly within expected bounds.

If those pass, the synthetic data is "good enough" for your MVP model and demo.

---

## 10. Economics & Prescriptive Parameters (Config-Only for MVP)

The prescriptive / optimisation layer in the MVP **does not introduce new
CSV datasets**.

All economic and decision parameters live in `config_mvp.yaml`, including:

- Penalty weights:
  - `costs.fresh.penalty_gap_per_unit_hour`
  - `costs.fuel.penalty_gap_per_trailer`
- Action costs:
  - `costs.fresh.bring_forward_per_unit`
  - `costs.fresh.air_per_unit`
  - `costs.fuel.bring_forward_per_unit`
  - `costs.fuel.air_per_unit` (may be unused / very high)
- Capacity / policy limits:
  - `shift_limits.fresh.max_forward_fraction`
  - `shift_limits.fresh.max_air_fraction`
  - `shift_limits.fuel.max_forward_fraction`
  - `shift_limits.fuel.max_air_fraction`
- Optional budgets:
  - `budgets.fresh_air_units_per_week`
  - `budgets.fresh_bring_forward_units_per_week`
  - (analogous keys for Fuel if needed)

The optimisation layer takes the **existing `E_loss` and exposure** implied by:

- `per_sailing_predictions.csv`
- `exposure_by_sailing.csv`

and chooses effective reduction fractions (`alpha` per category/week) via
a tiny grid-search optimiser. No extra columns are required in the synthetic
data for the MVP.

If we later want more heterogeneity (e.g. higher penalty for certain
"priority" sailings), we can optionally add:

- `priority_class` (enum) to `sailings.csv` or `exposure_by_sailing.csv`

but this is **not required** for the hackathon MVP.

No other SCHEMA sections need to change.

```

This keeps the schema tight, clearly separates **structure** from **synthesis logic**, and is realistic to implement in a few hours of coding.
::contentReference[oaicite:0]{index=0}
```
