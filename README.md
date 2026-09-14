# F1 2026 World Championship Prediction Model & Web App

An end-to-end, open-source machine-learning project for estimating Formula 1 race finishing positions and converting those forecasts into Drivers' Championship win probabilities through Monte Carlo simulation.

> **Important:** This is a forecasting/research project, not an official Formula 1/FIA model. It intentionally separates observed race data from 2026 technical-regulation regime features. Predictions are sensitive to data quality, missing variables, and assumptions.

## 1. Project overview

### Objectives

1. Ingest race data from the Jolpica F1 API with local CSV fallback.
2. Clean and validate driver, constructor, and result tables.
3. Build leakage-aware rolling-form and reliability features.
4. Encode the 2026 technical-regulation regime as model features.
5. Train a Scikit-Learn RandomForest regression model for race finishing position.
6. Simulate remaining races thousands of times.
7. Estimate Drivers' Championship win/podium probabilities.
8. Expose the workflow through a Streamlit dashboard.

### 2026 regulation features

The model includes regime descriptors for:

- Active aerodynamics / Straight and Corner modes.
- Overtake Mode replacing legacy DRS.
- Increased electrical contribution to the power unit.
- 400 kW ICE and approximately 350 kW electrical power.
- Removal of MGU-H.
- Advanced sustainable fuel.
- Lighter, narrower 2026 cars.
- Reduced drag/downforce targets.

These are **not** direct measurements of each team's engineering advantage. They are contextual features and should be replaced or supplemented with team/circuit telemetry, qualifying pace, tyre degradation, weather, and sector-speed data for a high-fidelity model.

## 2. Architecture

```text
                 ┌─────────────────────┐
                 │ Jolpica F1 API      │
                 │ /ergast/f1/...      │
                 └──────────┬──────────┘
                            │ fallback
                 ┌──────────▼──────────┐
                 │ CSV data/           │
                 │ results/drivers/... │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Cleaning + schema   │
                 │ validation          │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Feature engineering │
                 │ rolling form        │
                 │ reliability         │
                 │ grid factor         │
                 │ 2026 rules          │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │ RandomForestRegressor│
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Monte Carlo engine  │
                 │ 5,000+ simulations   │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Streamlit dashboard │
                 │ probabilities       │
                 │ driver deep dive    │
                 └─────────────────────┘
```

## 3. Repository layout

```text
f1-2026-prediction/
├── app.py
├── model_backend.py
├── requirements.txt
├── README.md
├── data/
│   ├── results.csv
│   ├── drivers.csv
│   ├── constructors.csv
│   └── schedule.csv              # optional fallback
├── notebooks/
│   └── f1_2026_predictor.ipynb
└── .gitignore
```

## 4. Tech stack

- Python 3.11+
- Pandas / NumPy
- Scikit-Learn
- Streamlit
- Altair
- Requests
- Jupyter
- Jolpica F1 API

Jolpica is the open-source successor to the Ergast F1 API and exposes Ergast-compatible routes such as `/ergast/f1/{season}/results/`, `/drivers/`, `/constructors/`, and `/races/`.

## 5. Installation

```bash
git clone <your-repository-url>
cd f1-2026-prediction

python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

## 6. Run the notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/f1_2026_predictor.ipynb
```

Or:

```bash
jupyter lab
```

## 7. Run the Streamlit app

```bash
streamlit run app.py
```

The app will:

- load 2026 data,
- train the model,
- infer completed/remaining rounds,
- run the configured Monte Carlo simulation,
- display championship probabilities.

## 8. CSV fallback

If the API is unavailable, place these files in `data/`:

```text
results.csv
drivers.csv
constructors.csv
```

Optional:

```text
schedule.csv
```

The backend automatically validates required columns before training.

## 9. Dataset schema

### results.csv

Required:

| Column | Type | Description |
| --- | --- | --- |
| season | int | Championship season |
| round | int | Race round |
| race_name | string | Grand Prix name |
| date | date | Race date |
| driver_id | string | Stable driver identifier |
| driver_code | string | Three-letter code |
| driver_name | string | Display name |
| constructor_id | string | Stable constructor identifier |
| constructor_name | string | Constructor/team name |
| grid | int | Starting grid |
| position | int | Classified finishing position |
| points | float | Race points |
| laps | int | Completed laps |
| status | string | Classified/DNF status |

Optional:

```text
fastest_lap_rank
fastest_lap_time
```

### drivers.csv

Required:

```text
driver_id
```

Recommended:

```text
code
given_name
family_name
date_of_birth
nationality
```

### constructors.csv

Required:

```text
constructor_id
```

Recommended:

```text
constructor_name
nationality
```

### schedule.csv

Recommended:

```text
season
round
race_name
circuit_id
circuit_name
date
time
```

## 10. Validation snippet

```python
import pandas as pd

REQUIRED = {
    "results.csv": {
        "season", "round", "driver_id", "constructor_id",
        "grid", "position", "points"
    },
    "drivers.csv": {"driver_id"},
    "constructors.csv": {"constructor_id"},
}

def validate_uploaded_data(results, drivers, constructors):
    frames = {
        "results.csv": results,
        "drivers.csv": drivers,
        "constructors.csv": constructors,
    }

    errors = []

    for filename, required in REQUIRED.items():
        missing = required - set(frames[filename].columns)
        if missing:
            errors.append(
                f"{filename}: missing columns {sorted(missing)}"
            )

        if frames[filename].empty:
            errors.append(f"{filename}: file is empty")

    if not results.empty:
        if results["driver_id"].isna().any():
            errors.append("results.csv: null driver_id values")
        if results["constructor_id"].isna().any():
            errors.append("results.csv: null constructor_id values")

        for col in ["season", "round", "grid", "position", "points"]:
            if col in results:
                converted = pd.to_numeric(results[col], errors="coerce")
                if converted.isna().any():
                    errors.append(f"results.csv: non-numeric values in {col}")

    if errors:
        raise ValueError("\\n".join(errors))

    return True
```

## 11. Modeling methodology

### Target

The supervised target is race finishing position.

Lower is better:

```text
P1 = 1
P2 = 2
...
P20 = 20
```

### Features

- Starting grid.
- Driver rolling finishing position over the previous 3 races.
- Driver rolling finishing position over the previous 5 races.
- Driver rolling points.
- Driver recent DNF rate.
- Constructor recent finishing-form.
- Constructor recent DNF/reliability rate.
- Grid-position factor.
- 2026 technical-regulation regime variables.
- Championship round progress.

All rolling features use `shift(1)` before rolling aggregation to avoid target leakage.

### Model

Default model:

```text
SimpleImputer(strategy="median")
        ↓
RandomForestRegressor
        ↓
race finishing-position estimate
```

Random Forest was selected for a transparent, robust baseline. A production research branch could compare:

- HistGradientBoostingRegressor
- XGBoost
- LightGBM
- CatBoost
- Bayesian hierarchical models
- ranking models / pairwise ranking

## 12. Monte Carlo simulation

For each remaining race:

1. Predict each driver's expected finishing position.
2. Apply reliability penalty.
3. Add stochastic uncertainty.
4. Rank drivers into a race classification.
5. Apply DNF probability.
6. Convert finishing positions to race points.
7. Repeat for every remaining round.
8. Repeat the entire season thousands of times.
9. Count how often each driver finishes P1 in the simulated championship.

Default:

```text
5,000 simulations
```

The Streamlit app permits 1,000–20,000 simulations.

## 13. Production improvements

For a serious predictive system, add:

- Qualifying lap-time pace.
- Sector-level speed.
- Circuit-specific driver/team effects.
- Tyre compound and degradation.
- Weather and track temperature.
- Safety-car probability.
- Pit-stop loss distribution.
- Sprint-weekend points.
- Penalties and grid drops.
- Team upgrades/change points.
- Power-unit component age.
- Historical driver-vs-teammate performance.
- Bayesian uncertainty around team strength.
- Time-aware cross-validation.
- Model calibration.
- Feature drift monitoring.
- Experiment tracking.
- Automated daily/weekly retraining.
- Unit/integration tests and CI.

Most importantly, **do not randomly split time-series race data** for final evaluation. Use walk-forward/time-aware validation.

## 14. Deployment

### Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new Streamlit app.
3. Select `app.py` as the entry point.
4. Use `requirements.txt`.
5. Deploy.

Recommended environment variables can include:

```text
F1_API_USER_AGENT=F1-2026-Prediction-Model/1.0
```

### Hugging Face Spaces

Create a Streamlit Space and upload:

```text
app.py
model_backend.py
requirements.txt
README.md
```

If using a committed CSV dataset, also upload `data/`.

For reproducibility, pin package versions in `requirements.txt`.

## 15. API notes

Jolpica requests should use a descriptive User-Agent. The project uses:

```text
F1-2026-Prediction-Model/1.0
```

The API is queried through the Ergast-compatible base route:

```text
https://api.jolpi.ca/ergast/f1/
```

The backend uses pagination with `limit=100`.

## 16. Regulatory source notes

The 2026 feature design is based on current FIA/F1 technical-regulation material. The repository should be periodically audited against the latest FIA-issued regulation documents because amendments can occur during a season.

Key concepts represented include active aerodynamics, Overtake Mode, increased electrical contribution, removal of MGU-H, sustainable fuel, and the lighter/narrower chassis concept.

## 17. License

MIT License is recommended for this repository. Verify that any third-party datasets you redistribute are compatible with their individual licenses and terms.

## 18. Disclaimer

Formula 1, FIA, teams, drivers, logos, trademarks, and related intellectual property belong to their respective owners. This independent project is not affiliated with or endorsed by Formula 1, the FIA, or any F1 team.
