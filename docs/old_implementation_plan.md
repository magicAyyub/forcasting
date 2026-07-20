# Verification Volume Forecasting and Ratio Simulation

This project aims to build a predictive modeling and simulation framework for daily verification volumes (`t_verified_canal_sum`) across different verification channels (`digital` and `bureau`). The model will capture seasonality, holiday/weekend effects, and the impact of service availability (`Disponibilite_de_service`), while simulating a planned policy shift in the channel ratio after October 2026.

## Data Science Insights (Exploratory Findings)

Based on a preliminary inspection of the `Acquisition_LIN.csv` dataset, here are the key characteristics that shape our modeling strategy:

1. **Duplicate Entries per Day/Channel**: The dataset has multiple rows for the same date and channel (e.g., two rows for `digital` on most days, one with a very low volume and another with a high volume). Since all day-level attributes (`ferie`, `week_end`, `Disponibilite_de_service`) are identical across these duplicates, we propose **aggregating (summing)** the volume per date and channel.
2. **Channel Volume & Relevance**:
   - `facteur` is inactive after 2022 (can be dropped).
   - `autre` has negligible volume in recent years (less than 200 verifications in 2025 and 2026 combined) and will be omitted or kept as a constant near-zero baseline.
   - `digital` and `bureau` represent the core channels, holding an overall historical ratio of **~72.3% Digital / 27.7% Bureau** during 2025-2026 (very close to the user's 75/25 estimate).
3. **Day-Type Confounding on Ratios**:
   - Although the overall ratio is ~72.3%, it varies significantly by day type:
     - **Regular Weekdays**: 69.2% Digital / 30.8% Bureau
     - **Weekends**: 86.4% Digital / 13.6% Bureau
     - **Holidays**: 99.4% Digital / 0.6% Bureau (due to physical office closures)
   - Therefore, a naive top-down model that splits *every day's* volume by a constant 75/25 or 80/20 ratio would be highly inaccurate on weekends and holidays.
4. **Availability Impact**:
   - `Disponibilite_de_service` is completely missing (`NaN`) in 2022, but fully populated with binary `0.0` or `1.0` values from 2023 onwards.
   - Days with `Disponibilite_de_service = 1.0` correspond to a significant volume drop on weekdays:
     - **Digital Volume** drops by **-8.2%** on weekdays.
     - **Bureau Volume** drops by **-16.7%** on weekdays.
   - This indicates that `1.0` represents a service disruption or outage, while `0.0` represents normal available service.
5. **Seasonality**:
   - There is strong monthly seasonality. For example, summer months (May, July, and August) see a **15% to 28% drop** in volume relative to the annual mean, while winter months (November, January) see a **15% to 23% increase**. This matches the user's observation of +/-10% activity fluctuations.

---

## User Review Required

> [!IMPORTANT]
> We must decide how to handle the ratio change after October 2026. Since the target overall shares (75/25, then 77/23, then 80/20) are business-wide targets, and since bureau offices are closed on weekends/holidays, we cannot apply the ratio uniformly to every day.
> **Our Proposed Solution**: We will map the overall weekly target ratios (75%, 77%, 80%) to corresponding **weekday-only targets** using the historical day-type volume weights. Our mathematical derivation shows:
> - Overall target **75.0%** digital $\rightarrow$ Weekday digital share should be set to **72.48%**.
> - Overall target **77.0%** digital $\rightarrow$ Weekday digital share should be set to **74.89%**.
> - Overall target **80.0%** digital $\rightarrow$ Weekday digital share should be set to **78.49%**.
>
> On weekends and holidays, we will keep the historical ratios (~86.4% and ~99.4% respectively) constant, as the physical channel cannot easily scale up on days when offices are closed.

---

## Open Questions

> [!QUESTION]
> 1. **Forecast Horizon**: How far into the future would you like to forecast? (e.g., until December 2026, or through the end of 2027?)
> 2. **Availability Imputation**: Since 2022 has no availability data, should we assume `Disponibilite_de_service = 0.0` (fully available) for all days in 2022 during training, or should we train the availability effect exclusively on 2023-2026 data?
> 3. **Future Availability Modeling**: When forecasting into the future (beyond June 2026), service outages are unknown. Should we assume a baseline outage probability (e.g., ~8% of weekdays based on history) or allow simulating specific scenarios (e.g., "0 outages" vs. "heavy outages")?

---

## Proposed Changes

We will create a structured, modular forecasting codebase in the workspace.

### Preprocessing and Feature Engineering

#### [NEW] [preprocess.py](file:///Users/ayouba/Documents/lin_projects/forcasting/preprocess.py)
This script will:
- Load `Acquisition_LIN.csv`.
- Filter out `facteur` and `autre`.
- Aggregate duplicates by summing `t_verified_canal_sum` grouped by `date` and `canal_verification`.
- Impute missing `Disponibilite_de_service` for 2022 with `0.0` (fully available) or handle as specified by the user.
- Engineer calendar features: `month_of_year`, `day_of_week`, `ferie`, `week_end`, and lag features for volume tracking.

---

### Modeling and Evaluation

#### [NEW] [model.py](file:///Users/ayouba/Documents/lin_projects/forcasting/model.py)
This script will implement the regression model.
- We propose using a Ridge Regression or a Generalized Linear Model (GLM) with Poisson/Negative Binomial link, or LightGBM, as it is robust for volume data, handles calendar indicators, and isolates the effect of `Disponibilite_de_service`.
- The model will forecast daily potential volumes for weekdays, weekends, and holidays.

#### [NEW] [evaluate.py](file:///Users/ayouba/Documents/lin_projects/forcasting/evaluate.py)
This script will evaluate the model performance.
- Perform a time-series walk-forward validation (e.g., using `TimeSeriesSplit` on historical data).
- Measure Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE).

---

### Simulation and Visualization

#### [NEW] [forecast_simulation.py](file:///Users/ayouba/Documents/lin_projects/forcasting/forecast_simulation.py)
This script will generate the future projections:
- Generate daily forecasts from June 2026 to the desired horizon.
- Apply the ratio transition plan:
  - Up to October 2026: ~75% overall digital share.
  - From November 2026: Transitioning to 77% digital.
  - Subsequently: Reaching 80% digital.
- Simulate two scenarios for service availability:
  - **Scenario A (Optimal)**: 100% service availability (no outages).
  - **Scenario B (Standard)**: Historical rate of outages (~8.7% of weekdays).

#### [NEW] [visualize.py](file:///Users/ayouba/Documents/lin_projects/forcasting/visualize.py)
A script that generates visual charts and exports them as artifacts:
- Historical vs. Forecasted Daily/Monthly volumes.
- Scenario comparison (Optimal vs. Standard availability).
- Digital vs. Bureau ratio transitions over time.

---

## Verification Plan

### Automated Tests
- We will run the python scripts sequentially:
  `python3 preprocess.py`
  `python3 evaluate.py`
  `python3 forecast_simulation.py`
- We will output metrics to verify that the cross-validation error is within acceptable bounds (MAPE < 15%) and that the forecasted channel ratios exactly match the target 75%, 77%, and 80% marks.

### Manual Verification
- We will inspect the generated plots and data tables (exported to the artifacts directory) to ensure the seasonality (9-10% changes) and outage impacts (-8.2% digital, -16.7% bureau) are correctly captured.
