# MILESTONES.md — IslandSense (Hackathon Build)

> **Goal:** Ship a *real* end-to-end prototype that proves the spine:
> **Per-sailing physics → daily JDI (7 days) → weekly JDI + optimised scenarios → hours/trailers avoided → per-sailing drilldown.**
> 24 hours of actual build time, within a 48-hour hackathon window.

This plan is **implementation-first**. Every milestone ends with *something that runs*.

---

## 0. Timeline at a Glance

> You do **not** have 48h of coding. Assume ~24h net build time.

| Phase     | Wall-clock window | Focus                                            |
|-----------|-------------------|--------------------------------------------------|
| **M0–M1** | H0–H6            | Data, labels, features, synthetic exposure       |
| **M1.5**  | H6–H7            | Economics & constraints (synthetic)              |
| **M2**    | H7–H11           | Per-sailing model + basic reliability            |
| **M2.5**  | H11–H13          | Tiny optimiser → prescriptive scenario           |
| **M3**    | H13–H17          | 7-day daily JDI + weekly scenarios & impact      |
| **M4**    | H17–H21          | Streamlit vertical slice (weekly + strip + drilldown) |
| **M5**    | H21–H24          | Demo hardening + slides                          |

If you slip, **protect M2–M4 at all costs**.
An ugly but working spine beats a beautiful half-product.

---

## M0 — Repo, Skeleton, and "Hello Sailings" (H0–H2)

**Objective:** Have a runnable repo with synthetic sailings flowing end-to-end into a basic per-sailing table.

### Tasks

- [x] Create repo structure:
  - `islandsense/`
    - `data/` (raw, synth, interim)
    - `src/`
      - `schema.py`
      - `synth.py`
      - `features.py`
      - `model.py`
      - `aggregate.py`
      - `app.py` (Streamlit stub)
    - `config.yaml`
    - `BRIEF_MVP.md`
    - `SPEC.md`
    - `MILESTONES.md`
    - `LIVING_ADR.md`
- [x] Implement **schema stubs** in `schema.py`:
  - Simple dataclasses or just helpers for:
    - `Sailing`, `Status`, `Metocean`, `Tide`, `ExposureBySailing`.
- [x] Write a tiny `synth.py` that generates:
  - `data/sailings.csv` (few routes, vessels, 7-day horizon).
  - `data/status.csv` with plausible delays/cancellations.
  - `data/metocean.csv`, `data/tides.csv` with plausible weather/tides.
  - `data/exposure_by_sailing.csv` with `fresh_units`, `fuel_units`.

### Deliverables

- `python -m islandsense.synth` produces CSVs in `data/`.
- A 10-line script (or notebook cell) loads `sailings.csv` and prints **5 sample rows**.

### Cut rules

If you're behind at M0:

- Skip dataclasses; just use `pandas` + `assert_required_columns(df, expected)`.

---

## M1 — Labels, Features & Synthetic Exposure (H2–H6)

**Objective:** Deterministic per-sailing feature matrix + labels + exposure per category (Fresh/Fuel).

### Tasks

- [x] In `features.py`:
  - Implement label function:
    - `disruption = (status == "cancelled") or (delay_min > 120)`
      (threshold from `config.yaml`).
  - Implement physics features:
    - `wotdi` (wind-heading misalignment × wind strength),
    - `bsef` (beam-sea exposure factor),
    - `gust_max_3h`,
    - `tide_gate_margin` (simple proxy),
    - `prior_24h_delay`,
    - `day_of_week`, `month`.
- [x] In `synth.py`:
  - Ensure `exposure_by_sailing.csv`:
    - Each sailing has `fresh_units` and `fuel_units`.
    - Keep totals per day roughly stable so JDIs look sensible.
- [x] Implement feature computation:
  - Load all CSVs.
  - Join into a **per-sailing DataFrame** with:
    - `sailing_id, route, vessel, etd_iso, features..., label, fresh_units, fuel_units`.

### Deliverables

- Feature computation runs and prints:
  - Feature columns & label distribution.
  - Sum of `fresh_units` and `fuel_units` per day.

### Cut rules

If behind:

- Drop `prior_24h_delay` and `tide_gate_margin` initially.
- Keep **wotdi, bsef, gust_max_3h, day_of_week, month** as the first-wave feature set.

---

## M1.5 — Synthetic Economics (Config-Only) *(no new CSVs)* (H6–H7) [STRETCH]

**Objective:** Hard-code a tiny synthetic economics model in `config_mvp.yaml`
so the optimiser can talk about "cost vs benefit" without changing any schemas.

### Tasks

- [x] **Config: costs & penalties**

  Add to `config_mvp.yaml`:

  ```yaml
  costs:
    fresh:
      penalty_gap_per_unit_hour: 10.0      # £ per pallet-hour of shelf gap
      bring_forward_per_unit:       1.0    # £ per pallet brought forward
      air_per_unit:                 4.0    # £ per pallet air-lifted
    fuel:
      penalty_gap_per_trailer:     50.0    # £ per trailer-equivalent at risk
      bring_forward_per_unit:       2.0
      air_per_unit:                10.0    # may be unused or set very high
  ```

- [x] **Config: shift limits & budgets** (fractions, not per-sailing caps)

  ```yaml
  shift_limits:
    fresh:
      max_forward_fraction: 0.30
      max_air_fraction:     0.15
    fuel:
      max_forward_fraction: 0.30
      max_air_fraction:     0.10

  budgets:
    fresh_air_units_per_week:  200.0   # soft cap; can be generous
    fresh_forward_units_per_week: 600.0
    # (fuel budgets optional for MVP)
  ```

- [x] **Config: optimiser grid**

  ```yaml
  optimiser:
    grid_forward: [0.0, 0.10, 0.20, 0.30]
    grid_air:     [0.0, 0.05, 0.10, 0.15]
  ```

- [x] **Wire to optimiser** (later M2.5 / M3)

  The optimiser:
  - reads `E_loss_total[c]` from aggregation,
  - reads `costs.*`, `shift_limits.*`, `budgets.*`, `optimiser.*` from config,
  - searches over `(x_c, y_c)` in the grid,
  - picks the best `(x_c, y_c)` per category that maximises
    `penalty_avoided - action_cost`, subject to shift_limits and budgets,
  - produces effective `alpha_eff[c]` that feed into the scenarios.

### Removed (was M1.5.1)

#### 1.5.1 ~~Capacity per sailing~~ (Removed)

Per-sailing capacity limits are **not required** for the MVP optimiser.
The prescriptive layer operates on weekly fractions (`shift_limits`) applied
to total exposure, not per-sailing headroom. This keeps all economics in
`config_mvp.yaml` and avoids changing any CSV schemas.

### Deliverables

- All economic parameters exist in `config_mvp.yaml`.
- Config loads without error.

### Cut rules

If behind:

- Skip M1.5 entirely; use fixed scenario alphas in M3.
- Come back to this if time permits after M4.

---

## M2 — Per-Sailing Model + Basic Reliability (H7–H11)

**Objective:** Train **one per-sailing model** and prove it's not a toy (basic Brier/AUC).

### Tasks

- [x] In `model.py`:
  - Train a LightGBM/XGBoost classifier on:
    - Features from M1,
    - Label from M1.
  - Use `random_seed` from `config.yaml`.
  - Simple train/validation split (80/20 or by route).
  - Compute:
    - Global Brier score,
    - AUC,
    - Optional simple reliability bins (ECE-lite).
- [x] Persist:
  - `models/model.pkl`
  - `models/calibrator.pkl`
  - `models/model_meta.json` with metrics & config snapshot.
- [x] Implement `predict.py`:
  - Load model + per-sailing features.
  - Output `per_sailing_predictions.csv` with:
    - `sailing_id, route, vessel, etd_iso, p_sail, fresh_units, fuel_units`.

### Deliverables

- `python train.py` produces model + metrics.
- `python predict.py --all` produces `per_sailing_predictions.csv`.
- A small printed summary of metrics in console.

### Cut rules

If behind:

- Skip fancy reliability; log just global Brier + AUC.
- Don't attempt per-route metrics in MVP.

---

## M2.5 — Tiny Optimiser → Prescriptive Scenario (H11–H13) [STRETCH]

**Objective:** Replace fixed scenario alphas with a tiny optimiser that maximises net benefit (penalty avoided - cost).

### Tasks

#### 2.5.1 Decision variables (per category)

- [ ] For each category `c ∈ {fresh, fuel}`:
  - `x_c` = weekly forward fraction (0…max_forward_fraction[c])
  - `y_c` = weekly air fraction (0…max_air_fraction[c])

#### 2.5.2 Objective function

- [ ] Compute:
  ```text
  E_loss_total[c] = Σ_d E_loss[c,d]
  E_loss_after[c](x_c, y_c) = E_loss_total[c] * (1 - beta_f * x_c - beta_a * y_c)

  penalty_avoided[c] = k_penalty[c] * (E_loss_total[c] - E_loss_after[c])
  total_cost[c] = cost_forward[c] * x_c * exposure[c] + cost_air[c] * y_c * exposure[c]

  NetBenefit[c] = penalty_avoided[c] - total_cost[c]
  ```

#### 2.5.3 Grid search

- [ ] Search over tiny grid:
  - `x_c ∈ {0.0, 0.1, 0.2, 0.3}`
  - `y_c ∈ {0.0, 0.05, 0.1}`
  - 12 combos per category, 144 total.

#### 2.5.4 Output effective alpha

- [ ] Back-solve `alpha_effective[c]` from chosen `(x_c, y_c)`:
  ```text
  alpha_effective[c] = beta_f * x_c + beta_a * y_c
  ```
- [ ] Write to `config_runtime.yaml` or inject into config:
  ```yaml
  scenarios:
    - id: scenario_A
      name: "Optimised weekly plan"
      alpha:
        fresh: <computed>
        fuel: <computed>
  ```

### Deliverables

- `python -m islandsense.optimizer`:
  - Prints chosen `x_c`, `y_c` per category.
  - Prints effective `alpha[c]`.
  - Prints net benefit in "£ saved vs £ spent".
- Scenario A in config is now computed, not hardcoded.

### Cut rules

If behind:

- Skip M2.5; use fixed alphas from config in M3.
- Keep as stretch goal after M4 working.

---

## M3 — 7-Day Daily JDI, Weekly JDI & Scenarios (H13–H17)

**Objective:** Convert per-sailing predictions into **daily JDI per category**, then into **weekly JDI**, and finally into **two simple weekly scenarios** with impact.

### Tasks

#### 3.1 Daily aggregation (in `aggregate.py`)

- [x] From `per_sailing_predictions.csv`:
  - Derive `day_index` for each sailing (D0..D6 from `now`),
  - Group by `(category, day_index)` where category ∈ {Fresh, Fuel}:
    - `E_loss[c,d] = Σ_s p_sail[s] * exposure_units[c,s]`.

- [x] Map to daily JDI (baseline):
  - Use `expected_loss_min[c]` and `expected_loss_max[c]` from `config.yaml`:
    - Normalise `E_loss[c,d]` to [0, 100] → `JDI_baseline[c,d]`.
  - Derive band (Green/Amber/Red) from config thresholds.

- [x] Compute weekly baseline JDI:
  - `weekly_JDI_baseline[c] = mean_d JDI_baseline[c,d]` (or simple average).

#### 3.2 Weekly scenarios (A/B)

- [x] Read scenario coefficients from config (or M2.5 optimizer output):

  ```yaml
  scenarios:
    - id: scenario_A
      name: "Bring forward 10%"
      alpha:
        fresh: 0.20
        fuel: 0.15
    - id: scenario_B
      name: "Bring forward 10% + air-lift 5%"
      alpha:
        fresh: 0.30
        fuel: 0.25
  ```

- [x] For each scenario `k`, category `c`, day `d`:
  ```text
  E_loss_k[c,d] = E_loss[c,d] * (1 - alpha_k[c])
  ```
- [x] Map `E_loss_k[c,d]` back to daily `JDI_k[c,d]` with same scaling.
- [x] Compute `weekly_JDI_k[c] = mean_d JDI_k[c,d]`.

#### 3.3 Impact calculation

- [x] From config:

  ```yaml
  impact:
    k_hours_per_unit: 0.25
    units_per_trailer: 24
  ```
- [x] For each scenario `k`, category `c`:

  ```text
  delta_E_loss[c,k] = Σ_d (E_loss[c,d] - E_loss_k[c,d])
  ```

  - Fresh: `hours_avoided[c,k] ≈ delta_E_loss[c,k] * k_hours_per_unit`
  - Fuel: `trailers_avoided[c,k] ≈ delta_E_loss[c,k] / units_per_trailer`

- [x] Write out:
  - `daily_jdi.csv` — `day_index, date, category, E_loss, JDI_baseline, band`.
  - `weekly_jdi.csv` — baseline + scenario JDI per category.
  - `scenario_impact.csv` — per scenario per category:
    - `weekly_JDI`, `hours_avoided`, `trailers_avoided`.

#### 3.4 Per-sailing contributions (for drilldown)

- [x] Compute per-sailing contributions:
  ```text
  contrib_fresh[s] = p_sail[s] * fresh_units[s]
  contrib_fuel[s]  = p_sail[s] * fuel_units[s]
  ```
- [x] Save `sailing_contrib.csv` with:
  - `sailing_id, day_index, date, route, vessel, etd_iso, p_sail, fresh_units, fuel_units, contrib_fresh, contrib_fuel`.

### Deliverables

- `python -m islandsense.aggregate` produces:
  - `daily_jdi.csv`
  - `weekly_jdi.csv`
  - `scenario_impact.csv`
  - `sailing_contrib.csv`

### Cut rules

If behind:

- Implement only **Scenario A** (recommended plan) and hard-code Scenario B values in UI copy.
- Skip per-day scenario JDIs; focus on weekly JDI + daily baseline JDI + per-sailing contributions.

---

## M4 — Streamlit Vertical Slice (H17–H21)

**Objective:** A *single-page app* that shows the whole story:

> Weekly JDI → one recommended weekly plan → 7-day strip → day & per-sailing drilldown.

### Tasks

- [ ] In `app.py` (Streamlit):

  - Load:
    - `weekly_jdi.csv`
    - `scenario_impact.csv`
    - `daily_jdi.csv`
    - `sailing_contrib.csv`
  - Define `now` and map `day_index` → actual dates for display.

- [ ] Layout:
  **Top row: Weekly JDI + scenarios**

  - Left: **Weekly JDI card**
    - For Fresh & Fuel:
      - Baseline JDI,
      - Recommended scenario JDI,
      - Δ points + hours/trailers avoided.
  - Right: **Scenario strip**
    - Large card for recommended scenario (Scenario A),
    - Small grey card for alternative scenario (Scenario B),
    - Clicking on recommended scenario opens scenario detail (optional).

  **Bottom row: 7-day daily JDI strip**

  - 7 tiles (D0..D6):
    - Date label,
    - Fresh JDI + band,
    - Fuel JDI + band.
  - Clicking a tile:
    - Shows **Day detail**:
      - Baseline JDIs for that day (Fresh/Fuel),
      - Simple text about how recommended plan applies,
      - Table of top N sailings for that day, sorted by `contrib_fresh` or `contrib_fuel`.

### Deliverables

- `streamlit run src/islandsense/app.py`:
  - Displays:
    - Weekly JDI + scenarios at top,
    - 7-day strip at bottom,
    - Working click → day drilldown with per-sailing table.

### Cut rules

If behind:

- Drop the alternative scenario card; only show the **recommended** scenario.
- Simplify drilldown:
  - Show per-sailing table for a single "worst day" instead of all 7 days.

---

## M5 — Demo Hardening & Slides (H21–H24)

**Objective:** Polish just enough that it feels intentional, not random.

### Tasks

- [ ] Add **basic "Why"** explanation for sailings in the day drilldown:
  - Show `wotdi` and `bsef` values as columns and one sentence:
    - e.g. "High beam-sea exposure + strong cross-wind".
- [ ] Add **static label** in UI for model quality:
  - "Trained on synthetic data — example metrics: Brier=0.xx, AUC=0.yy (validation)".
- [ ] Optional: **Export button**:
  - `st.download_button` to export `ops_brief_week.csv` or filtered views.
- [ ] Prepare a **3–5 slide deck** (or notes) that follow the spine:
  1. Jersey context & pain (Fresh/Fuel, mono-threaded supply).
  2. Per-sailing physics → disruption probability.
  3. Daily JDI strip (7-day) and what "red days" mean.
  4. Weekly JDI + recommended plan + hours/trailers avoided.
  5. Drilldown to specific sailings.

### Deliverables

- Simple `make demo` (or shell script) that runs:
  - synth → train → predict → aggregate → `streamlit run app.py`.
- You can talk through the prototype coherently in **3 minutes**.

### Cut rules

If out of time:

- Drop export; keep everything in the Streamlit UI.
- Don't chase per-feature explanation; just display `p_sail`, `fresh_units`, `fuel_units`.
- Make sure:
  - Model trains,
  - `daily_jdi.csv` and `weekly_jdi.csv` exist,
  - The app shows:
    - weekly JDI,
    - one scenario card,
    - 7-day strip,
    - one functioning day drilldown.

---

## "If I Only Have 8–10 Hours, What Do I Cut?"

**Must have (no compromise):**

1. Synthetic per-sailing data + features + labels (`synth.py`).
2. Working per-sailing model that outputs `p_sail` (M2).
3. Deterministic roll-up to:
   - daily JDI per category,
   - weekly baseline JDI (M3 core).
4. One weekly scenario (Scenario A) with:
   - weekly JDI reduction,
   - hours/trailers avoided.
5. A Streamlit page that shows:
   - Weekly JDI,
   - the recommended scenario,
   - a 7-day strip where at least one day tile drills into a per-sailing table.

**Nice-to-have only if time (priority order):**

1. M2.5: Optimiser that computes scenario alphas (makes judges go 🧐💡).
2. M1.5: Full economic constraints in config.
3. Alternative scenario card (Scenario B).
4. Full scenario detail view.
5. All 7 days wired up for drilldown.
6. Export CSV / PDF.
7. Fancy reliability metrics, SHAP, or anomaly banners.

---

## Quick Mapping Back to LIVING_ADR Decisions

- **D-001** (per-sailing physics) → M1, M2.
- **D-002** (physics-first, not black box) → M1 feature set + "Why" columns.
- **D-003** (JDI via algebra, not a second model) → M3 daily & weekly JDI.
- **D-004** (simple scenarios, no LP) → M3 scenario scaling.
- **D-004b** (optional prescriptive core) → M1.5 + M2.5 stretch goals.
- **D-005** (story spine) → M4 + M5 layout and pitch structure.

If a task doesn't clearly serve one of these, it's a **candidate to cut**.

---

## Current Progress Summary

- **M0**: ✅ Complete
- **M1**: ✅ Complete
- **M1.5**: ✅ Complete (economics in config.yaml)
- **M2**: ✅ Complete (model + calibrator + predict pipeline)
- **M2.5**: ⏸️ Stretch goal (skipped for now)
- **M3**: ✅ Complete (aggregate.py + unit tests)
- **M4**: 🔄 Next priority
- **M5**: ⏳ Pending
