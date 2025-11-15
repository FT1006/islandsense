Short answer:
Yes, the spec** ***does* need a rewrite, because it’s still hard-wired to:

* **72h rolling horizon** , not** ** **7-day daily forecast** ,
* **per-category 3 actions** (Green/Amber/Red → do nothing / bring forward / partial air),
* A** ****2-tile + what-if slider** UI instead of** ** **weekly JDI + scenarios + 7-day strip + drilldown** .

If you keep this SPEC as-is while your brief has moved to “7-day Early-Warning + weekly scenarios”, you’ll get spec drift and weird questions (“where’s the 72h view?”, “why only two tiles?”).

Below is a SPEC.md rewritten to match the new** ****7-day early-warning + weekly scenario + daily and per-sail drilldown** design, but keeping the same structure and “brutally minimal” intent.

---

```markdown
# SPEC.md — IslandSense MVP

**Name:** IslandSense — Per-Sailing Disruption & 7-Day Early Warning (MVP)  
**Scope:** 1 hackathon weekend, single-page prototype, simulated/seed data only.

---

## 1. Purpose & Problem

Jersey depends on a small number of sea sailings for:

- **Fresh food** (risk: empty shelves)
- **Fuel** (risk: disrupted depot operations)

Currently, decisions about delays, bring-forwards, or air-lifts are mostly reactive and qualitative.

There is **no small, shared tool** that:

1. Quantifies **disruption probability per sailing** using physics (wind, waves, tides),
2. Rolls that into **simple daily risk scores per category** (Fresh/Fuel) over the **next 7 days**, and
3. Suggests **one simple weekly plan** plus an estimate of **“hours / trailers avoided”**,
4. While still allowing planners to see **which specific sailings** drive that risk.

This MVP is that tool.

---

## 2. Goals (MVP)

In ~24 hours, we will:

1. Ingest seeded CSVs of scheduled sailings, status, weather, tides, and exposure.
2. Build a **per-sailing binary classification model**:
   - Input: physics-based features (wind/wave/tide vs vessel).
   - Output: disruption probability per sailing (`p_sail`).
3. Define a **7-day forecast horizon** starting from `now`:
   - Calendar days D0..D6 (today + next 6 days).
4. For each day and category (**Fresh/Fuel**):
   - Compute daily expected loss `E_loss[c,d]`,
   - Map to a **0–100 daily JDI** and band (Green/Amber/Red).
5. Compute **weekly baseline JDI** (Fresh & Fuel) from the 7 daily values.
6. Define two **weekly scenarios** (policies):
   - Scenario A: “Bring forward 10% of volume”
   - Scenario B: “Bring forward 10% + air-lift 5%”
7. Approximate the effect of each scenario as:
   - Reduced weekly JDI for Fresh & Fuel vs baseline,
   - **“~X hours of shelf-gap avoided”** (Fresh),
   - **“~Y trailer-equivalents at risk avoided”** (Fuel).
8. Choose **one recommended weekly scenario** (usually Scenario A).
9. Expose everything in a **single Streamlit page** with:
   - Weekly JDI card (Fresh/Fuel, baseline vs recommended),
   - Two scenario cards (recommended + alternative),
   - 7-day daily JDI strip (baseline),
   - Drilldown:
     - Day → daily detail + per-sailing list,
     - Scenario → weekly detail + top contributing sailings.
10. (Optional, stretch) Derive the coefficients for the recommended weekly scenario
    from a tiny optimisation step that trades off:
    - shelf-gap / shortfall penalty (Fresh/Fuel), and
    - costs of bring-forward and air-lift,
    rather than fixing them purely as heuristic alphas.

---

## 3. Non-goals (out of scope)

For this MVP we explicitly do **not**:

- Connect to any live PoJ / GoJ or metocean APIs.
- Model demand, stock levels, or inventory curves.
- Implement a full optimisation/budget LP for volume shifts.
- Support multiple custom horizons, rolling hours, or user-configured windows.
- Implement anomaly detection over freight stats or price proxies (we may fake columns but won’t use them).
- Build auth, multi-user, notifications, CI/CD, or production-grade monitoring.

Everything above is “next steps” material, not in this spec.

---

## 4. Architecture (very small)

One Python codebase, one process.

**Core scripts**

- `prep.py`  
  - Load CSVs, build training table, train model, save model artifact.
- `app.py` (Streamlit app)  
  - Load model + config + seeded data,
  - Run per-sailing inference for the next 7 days,
  - Aggregate to:
    - daily JDI per category,
    - weekly JDI baseline,
    - weekly JDI per scenario,
  - Choose recommended weekly scenario and compute impact,
  - Render UI (weekly + scenarios + 7-day strip + drilldown).

**Configuration**

- `config_mvp.yaml` — minimal configuration:
  - Forecast horizon (days),
  - JDI scaling/bands,
  - Scenario coefficients,
  - Impact constants,
  - Model seed, label thresholds.

**Artifacts**

- `model.pkl` — trained classifier.
- Optional:
  - `model_meta.json` — metrics and config snapshot.
  - `ops_brief_week.csv` — exported weekly brief.

---

## 5. Data model & files

All inputs are local CSVs. No external dependencies.

### 5.1 Files

1. `sailings.csv`
   - One row per scheduled sailing.
   - Required columns:
     - `sailing_id` (string)
     - `route` (string, e.g. "Portsmouth→Jersey")
     - `vessel` (string)
     - `etd_iso` (ISO 8601, UTC)
     - `eta_iso` (ISO 8601, UTC, optional)
     - `head_deg` (float, vessel heading on approach, degrees)

2. `status.csv`
   - One row per sailing with outcome.
   - Required columns:
     - `sailing_id` (string, joins to `sailings.csv`)
     - `status` (enum: "arrived" / "cancelled")
     - `delay_min` (integer, minutes; 0 if on time / missing)

3. `metocean.csv`
   - Time series of metocean conditions.
   - Required columns:
     - `ts_iso` (ISO 8601, UTC)
     - `wind_kts` (float)
     - `wind_dir_deg` (float)
     - `gust_kts` (float)
     - `hs_m` (float, significant wave height)
     - `tp_s` (float, peak wave period)
     - `wave_dir_deg` (float)

4. `tides.csv`
   - Time series of tide heights.
   - Required columns:
     - `ts_iso` (ISO 8601, UTC)
     - `tide_m` (float)

5. `exposure_by_sailing.csv` (can be synthetic)
   - Approximate exposure per sailing and per category.
   - Required columns:
     - `sailing_id` (string)
     - `fresh_units` (float, e.g. pallets)
     - `fuel_units` (float, e.g. trailer-equivalents)
   - For MVP you can synthesise values that look plausible.

### 5.2 Economics & constraints (config-based, no extra CSVs)

For the MVP, all economic parameters and simple constraints for the
(optional) prescriptive core live in `config_mvp.yaml`, not in separate
tables. We introduce:

- `costs`:
  - `costs.fresh.bring_forward_per_unit`
  - `costs.fresh.air_per_unit`
  - `costs.fresh.penalty_gap_per_unit_hour`
  - `costs.fuel.bring_forward_per_unit`
  - `costs.fuel.air_per_unit` (may be set very high or zero)
  - `costs.fuel.penalty_gap_per_trailer`

- `budgets` (optional, can be generous in the MVP):
  - `budgets.fresh_air_units_per_week`
  - `budgets.fresh_bring_forward_units_per_week`
  - (analogous keys for Fuel if used)

- `shift_limits`:
  - `shift_limits.fresh.max_forward_fraction`
  - `shift_limits.fresh.max_air_fraction`
  - `shift_limits.fuel.max_forward_fraction`
  - `shift_limits.fuel.max_air_fraction`

These parameters are only required if we enable the tiny prescriptive core.
If they are missing, the system falls back to static scenario alphas.

---

## 6. Label & features

### 6.1 Label (binary disruption)

For each `sailing_id` in the joined `sailings + status` table:

```text
disruption = 1 if (status == "cancelled") OR (delay_min > disruption_delay_minutes), else 0
```

* `disruption_delay_minutes` is read from** **`config_mvp.yaml` (default: 120).

### 6.2 Feature engineering (minimal)

For each sailing:

1. Align metocean/tide:

   * Select metocean & tide rows around** **`etd_iso` (nearest or ±1h window).
   * Compute mean/last values per sailing.
2. Physics-inspired features (same as brief):

   * Relative angles:
     * `rel_wind_deg = wrap_angle(wind_dir_deg - head_deg)` in [-180, 180]
     * `rel_wave_deg = wrap_angle(wave_dir_deg - head_deg)` in [-180, 180]
   * **WOTDI** (wind-heading misalignment × wind strength):
     ```text
     wotdi = |sin(radians(rel_wind_deg))| * (wind_kts / 20.0)
     ```
   * **BSEF** (beam-sea exposure factor):
     ```text
     bsef = abs(sin(radians(rel_wave_deg))) * hs_m
     ```
   * **Gust feature** :

   ```text
   gust_max_3h = max(gust_kts) over [ETD - 3h, ETD]
   ```

   * **Tide feature** (simple proxy):
     ```text
     tide_gate_margin = minutes until next low tide after ETD
     ```
   * **Historical feature** :

   ```text
   prior_24h_delay = mean delay (minutes) on same route in prior 24h
   ```

   * Temporal features:
     * `day_of_week` (0=Monday..6=Sunday)
     * `month` (1-12)
3. MVP feature list:

   * `wotdi`
   * `bsef`
   * `gust_max_3h`
   * `tide_gate_margin`
   * `prior_24h_delay`
   * `day_of_week`
   * `month`

We only add more features if strictly needed.

---

## 7. Model training

### 7.1 Model type

* `xgboost` or** **`lightgbm` gradient-boosted tree classifier.

### 7.2 Training process

1. Load training table (features + label).
2. Perform a simple train/validation split (e.g. 80/20).
3. Train with:
   * `random_seed` from** **`config_mvp.yaml` (default: 42).
4. Evaluate on validation:
   * Brier score (primary).
   * AUC (secondary).
   * Accuracy (tertiary).

### 7.3 Output

* Save trained model to** **`model.pkl`.
* Save basic metrics + config to** **`model_meta.json`.

### 7.4 Quality targets (soft)

On validation:

* Brier ≤ 0.070,
* ECE ≤ 0.05 if we have calibration (optional),
* AUC ≥ 0.75.

If not met on synthetic data, we still ship; UI labels show “Experimental (synthetic data)”.

---

## 8. Inference & aggregation

### 8.1 Per-sailing predictions (7-day horizon)

At app startup (`app.py`):

1. Load:

   * `config_mvp.yaml`
   * `model.pkl`
   * each CSV.
2. Define** **`now` (UTC or config).
3. Filter** **`sailings` to those with** **`etd_iso` in:

   ```text
   [now, now + horizon_days * 24h)
   ```

   * `horizon_days` from config (default: 7).
4. Build features for these sailings using the same function as training.
5. Run** **`model.predict_proba(X)` to get** **`p_sail` for each sailing.

Internal table:

* `sailing_id, route, vessel, etd_iso, etd_day (D0..D6), p_sail, fresh_units, fuel_units, [wotdi, bsef, ...]`

Optional: include simple “why” values (e.g. show** **`wotdi` and** **`bsef` in UI).

### 8.2 Daily category risk / JDI (7-day strip)

Categories:** **`"fresh"`,** **`"fuel"`.

For each category** **`c` and day** **`d`:

1. Compute expected loss:

   ```text
   E_loss[c,d] = Σ_{s in day d} (p_sail[s] * exposure_units[c,s])
   ```

   where:

   * `exposure_units["fresh", s] = fresh_units`,
   * `exposure_units["fuel", s] = fuel_units`.
2. Convert to JDI index:

   ```text
   E_norm = clamp((E_loss[c,d] - expected_loss_min[c]) /
                  (expected_loss_max[c] - expected_loss_min[c]), 0, 1)
   JDI[c,d] = round(E_norm * 100)
   ```

   * `expected_loss_min[c]`,** **`expected_loss_max[c]` from config.
3. Determine JDI band for that day:

   * `green`: [0, 39]
   * `amber`: [40, 69]
   * `red`: [70, 100] (configurable)

We then derive weekly baseline JDI per category, e.g.:

```text
weekly_JDI_baseline[c] = round(mean_d JDI[c,d])
```

(Simple average over the 7 days; can be changed later.)

---

## 9. Weekly scenarios & impact

We define two weekly scenarios in `config_mvp.yaml`. In the simplest MVP
version, each scenario has fixed `alpha` coefficients per category, chosen
by hand. In a stretch version, these `alpha` values can be **derived by a
small optimisation step** (see §9.3) that trades off penalty avoided vs
action cost, but the surrounding logic (JDI scaling and impact computation)
remains unchanged.

```yaml
scenarios:
  - id: "scenario_A"
    name: "Bring forward 10%"
    alpha:
      fresh: 0.20  # 20% effective reduction in E_loss for Fresh
      fuel: 0.15
  - id: "scenario_B"
    name: "Bring forward 10% + air-lift 5%"
    alpha:
      fresh: 0.30
      fuel: 0.25
```

For each scenario** **`k`, category** **`c`, day** **`d`:

```text
E_loss_k[c,d] = E_loss[c,d] * (1 - alpha_k[c])
```

Then:

* Map** **`E_loss_k[c,d]` to daily** **`JDI_k[c,d]` using the same scaling as baseline.
* Compute weekly JDI under each scenario:

```text
weekly_JDI_k[c] = round(mean_d JDI_k[c,d])
```

### 9.1 Choosing the recommended scenario

For MVP, we use a simple rule:

* Prefer Scenario A by default.
* If scenario B gives substantially better improvement for Fresh (e.g. ≥ Δ_threshold) and we want to demo “more aggressive” option, we show it as the** ****alternative** in grey.

We** ****always** show:

* Baseline weekly JDI (Fresh/Fuel),
* Weekly JDI under recommended scenario,
* Weekly JDI under the alternative scenario.

### 9.2 Impact in hours / trailers

From config:

```yaml
impact:
  k_hours_per_unit: 0.25       # hours of shelf-gap avoided per Fresh unit
  units_per_trailer: 24        # units per Fuel trailer-equivalent
```

For each scenario** **`k` and category** **`c`:

```text
delta_E_loss[c] = Σ_d (E_loss[c,d] - E_loss_k[c,d])
```

* **Fresh:**
  ```text
  hours_avoided[c,k] ≈ delta_E_loss[c] * k_hours_per_unit
  ```
* **Fuel:**
  ```text
  trailers_avoided[c,k] ≈ delta_E_loss[c] / units_per_trailer
  ```

Round to 1 decimal place for display.

### 9.3 Optional: Tiny prescriptive optimiser to set scenario alphas

In the stretch (prescriptive) version, instead of hard-coding `alpha`
values for Scenario A/B, we derive the `alpha` for the **recommended**
scenario from a small optimisation step.

At a minimum, for each category `c ∈ {fresh, fuel}` we consider:

- `x_c` = weekly forward fraction (0 … max_forward_fraction[c])
- `y_c` = weekly air fraction (0 … max_air_fraction[c])

We approximate:

- Baseline total expected loss:
  `E_loss_total[c] = Σ_d E_loss[c,d]` (over 7 days).
- Post-plan expected loss:
  `E_loss_after[c](x_c, y_c) = E_loss_total[c] * (1 - β_f[c] * x_c - β_a[c] * y_c)`
  with small coefficients `β_f`, `β_a` defined in `config_mvp.yaml`.

We then define, per category:

- Penalty avoided:
  `penalty_avoided[c](x_c, y_c) = penalty_gap_per_unit[c] × (E_loss_total[c] - E_loss_after[c](x_c, y_c))`
- Action cost:
  - `total_forward_units[c] = x_c × total_exposure_units[c]`
  - `total_air_units[c] = y_c × total_exposure_units[c]`
  - `cost_forward[c] = bring_forward_per_unit[c] × total_forward_units[c]`
  - `cost_air[c] = air_per_unit[c] × total_air_units[c]`
  - `action_cost[c] = cost_forward[c] + cost_air[c]`

We search over a small grid of `(x_c, y_c)` values (e.g. 0, 0.1, 0.2, 0.3)
subject to budget and shift limits, and select the combination that
maximises:

```text
NetBenefit_total = Σ_c (penalty_avoided[c] - action_cost[c])
```

From the chosen `(x_c, y_c)`, we back out an effective reduction fraction
`alpha_eff[c]` and use that as:

```yaml
scenarios:
  - id: "scenario_A"
    name: "Optimised weekly plan"
    alpha:
      fresh: alpha_eff["fresh"]
      fuel:  alpha_eff["fuel"]
```

The rest of the pipeline (daily scenario JDIs, weekly JDI per scenario,
hours/trailers avoided) is unchanged.

If this optimiser is not implemented or disabled, we simply keep the
hand-tuned alpha values from the static config and still satisfy the
MVP spec.

---

## 10. UI spec (Streamlit)

Single page, no routing.

### 10.1 Top row — Weekly JDI + scenario cards

**Left: Weekly JDI card**

* Title: “This week’s risk — Fresh & Fuel”
* For** ** **Fresh** :
  * “Baseline JDI: X”
  * “With recommended plan: Y”
  * “Δ: –(X–Y) points, ~hours_avoided[Fresh, recommended] h avoided”
* For** ** **Fuel** :
  * Same but with trailers avoided.

**Right: Scenario strip (2 stacked cards)**

* **Large card (recommended scenario)** :
* Name: “Recommended weekly plan: Bring forward 10%”
* Bullets:
  * “Fresh: 57 → 39 (–18)”
  * “Fuel: 34 → 27 (–7)”
  * “Avoided: ~9 hours shelf-gap, ~3 trailers at risk”
* Click → scenario detail modal/panel.
* **Smaller grey card (alternative scenario)** :
* Name: “Alternative: Bring forward 10% + air-lift 5%”
* Short summary of impact.

### 10.2 Bottom row — 7-day daily JDI strip

* 7 tiles (carousel if needed), one per day D0..D6.
* Each tile:
  * Date label: “Tue 18”
  * Fresh: small dot + “F: 82 (Red)”
  * Fuel: small dot + “Fu: 61 (Amber)”
* Caption:
  > “Daily JDI forecast (baseline / do nothing). Click a day to see detail and which sailings drive the risk.”
  >

**Click a tile → Day detail**

* Show:
  * “Tue 18 — Fresh: RED 82, Fuel: AMBER 61 (baseline).”
  * Under recommended weekly plan:
    * Simple text:
      * “Fresh: apply bring-forward to highest-risk sailings.”
      * “Fuel: monitor (no extra moves).”
* Below:** ****per-sailing table (that day)**
  * `route, vessel, etd_iso, p_sail, fresh_units, fuel_units, contrib_fresh, contrib_fuel`.

Highlight top N by contribution.

### 10.3 Scenario detail (optional modal/panel)

* Show:
  * Weekly JDI baseline vs scenario (Fresh/Fuel),
  * Per-day deltas (e.g. a small table of** **`day, JDI_baseline, JDI_scenario`),
  * Top N sailings across the week with largest** **`delta contrib`.

---

## 11. Ops Brief (optional)

If time allows, implement a simple CSV export:

* `ops_brief_week.csv` with:
  * Header:
    * Generated at** **`<UTC timestamp>`
    * Forecast horizon: next 7 days
  * Summary rows:
    * Category, weekly JDI baseline, weekly JDI recommended, hours/trailers avoided.
  * Daily rows:
    * Day, category, JDI_baseline, JDI_recommended.
  * Per-sailing rows (top N):
    * `day, sailing_id, route, etd_iso, p_sail, fresh_units, fuel_units`.

This is a** ** **nice-to-have** , not essential for MVP acceptance.

---

## 12. Edge cases / assumptions

* If** **`exposure_by_sailing.csv` has no row for a sailing:
  * treat** **`fresh_units=fuel_units=0`.
* If** **`E_loss[c,d]` >** **`expected_loss_max[c]`:
  * clamp JDI to 100.
* If** **`E_loss[c,d]` <** **`expected_loss_min[c]`:
  * clamp JDI to 0.
* If** ****no sailings** in the next 7 days:
  * weekly JDI = 0, all days = Green, UI shows “No relevant sailings in forecast window.”
* If model prediction fails for a sailing:
  * log error; default** **`p_sail=0` for that sailing.

---

## 13. Acceptance checklist (MVP)

You’re done when:

1. **Model & data**
   * `prep.py` runs on seeded CSVs and produces** **`model.pkl` and** **`model_meta.json`.
   * You can print at least one validation metric (e.g. Brier score).
2. **Core logic**
   * `app.py`:
     * Computes** **`p_sail` for all sailings in the next 7 days,
     * Computes** **`E_loss[c,d]` and** **`JDI[c,d]` for each category & day,
     * Computes weekly baseline JDI,
     * Computes weekly JDI under scenarios A/B,
     * Computes hours/trailers avoided per scenario.
3. **UI**
   * Top row:
     * Weekly JDI card (Fresh/Fuel, baseline vs recommended),
     * Two scenario cards (recommended big, alternative small and grey).
   * Bottom row:
     * 7-day daily JDI strip (baseline),
     * Clicking a day shows daily detail + per-sailing table.
4. **Scenario logic**
   * Recommended scenario chosen deterministically based on config and weekly improvement,
   * Numbers on scenario cards match backend computations.
   * If the prescriptive optimiser is enabled, the recommended scenario's alphas
     must match the optimiser output; otherwise, static config alphas are used
     and this still satisfies the MVP.
5. **Determinism**
   * Given the same CSVs, config, and code:
     * Two runs produce the same per-sailing predictions, daily JDIs, and weekly scenario results (model seed fixed).

If all above are true, the spec is satisfied for the 7-day early-warning MVP.

```

If you want, I can also give you a **delta diff** (what to change vs your original SPEC) instead of a full replacement, but given how much the horizon + UI story shifted, I’d lean toward dropping this whole new SPEC in and deleting the 72h-specific one.
```
