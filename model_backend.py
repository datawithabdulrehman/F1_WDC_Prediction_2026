"""
F1 2026 World Championship Prediction Engine
=============================================

Production-oriented, dependency-light backend for:
- Jolpica/Ergast-compatible ingestion
- CSV fallback ingestion
- Feature engineering
- RandomForest regression for finishing-position prediction
- Monte Carlo championship simulation

The model is intentionally transparent: it predicts finishing position, then the
simulation converts sampled finishing positions into official race points.

2026 rules are represented as model features, not as a claim that the regulations
alone determine performance. Current regulatory constants should be reviewed against
the latest FIA technical regulations before a production retrain.
"""

from __future__ import annotations

import io
import logging
import math
import os
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

LOGGER = logging.getLogger(__name__)

JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1"
DEFAULT_USER_AGENT = "F1-2026-Prediction-Model/1.0"

# 2026 regulation-derived constants. These are features representing the new
# technical regime; they are not direct performance measurements.
RULES_2026 = {
    "active_aero": 1.0,
    "overtake_mode": 1.0,
    "drs_legacy": 0.0,
    "mgu_h_present": 0.0,
    "electric_power_kw": 350.0,
    "ice_power_kw": 400.0,
    "electric_power_share": 350.0 / (350.0 + 400.0),
    "sustainable_fuel": 1.0,
    "minimum_car_mass_kg": 768.0,
    "wheelbase_mm": 3400.0,
    "car_width_mm": 1900.0,
    "front_tyre_width_reduction_mm": 25.0,
    "rear_tyre_width_reduction_mm": 30.0,
    "downforce_reduction_pct": 30.0,
    "drag_reduction_pct": 55.0,
}

# Standard F1 race points, excluding fastest-lap bonus (removed from the system
# after 2024). Sprint points are deliberately not included in the core race
# simulation because the requested target is race finishing outcomes.
RACE_POINTS = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1
}


@dataclass
class F1DataBundle:
    results: pd.DataFrame
    drivers: pd.DataFrame
    constructors: pd.DataFrame
    schedule: pd.DataFrame | None = None


def _get_json(url: str, timeout: int = 30) -> dict:
    headers = {
        "User-Agent": os.getenv("F1_API_USER_AGENT", DEFAULT_USER_AGENT),
        "Accept": "application/json",
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _ergast_table(payload: Mapping) -> list[dict]:
    """Extract a Jolpica Ergast-compatible table from JSON."""
    mr = payload.get("MRData", payload)
    table = mr.get("RaceTable", mr.get("StandingsTable", {}))
    races = table.get("Races", [])
    return races


def _flatten_results_payload(payload: Mapping) -> pd.DataFrame:
    rows: list[dict] = []
    for race in _ergast_table(payload):
        season = race.get("season")
        rnd = race.get("round")
        race_name = race.get("raceName")
        race_date = race.get("date")
        for result in race.get("Results", []):
            driver = result.get("Driver", {})
            constructor = result.get("Constructor", {})
            fastest = result.get("FastestLap", {})
            rows.append({
                "season": int(season) if season is not None else np.nan,
                "round": int(rnd) if rnd is not None else np.nan,
                "race_name": race_name,
                "date": race_date,
                "driver_id": driver.get("driverId"),
                "driver_code": driver.get("code"),
                "driver_name": " ".join(
                    x for x in [driver.get("givenName"), driver.get("familyName")] if x
                ),
                "constructor_id": constructor.get("constructorId"),
                "constructor_name": constructor.get("name"),
                "grid": result.get("grid"),
                "position": result.get("position"),
                "points": result.get("points"),
                "laps": result.get("laps"),
                "status": result.get("status"),
                "fastest_lap_rank": fastest.get("rank"),
                "fastest_lap_time": (fastest.get("AverageSpeed") or {}).get("speed")
                if isinstance(fastest.get("AverageSpeed"), dict) else None,
            })
    return pd.DataFrame(rows)


def fetch_jolpica_results(season: int | str = "current") -> pd.DataFrame:
    """Fetch all available race results for a season, handling API pagination."""
    frames = []
    offset = 0
    limit = 100
    while True:
        url = f"{JOLPICA_BASE_URL}/{season}/results.json?limit={limit}&offset={offset}"
        payload = _get_json(url)
        frame = _flatten_results_payload(payload)
        if not frame.empty:
            frames.append(frame)
        total = int(payload.get("MRData", {}).get("total", len(frame)))
        offset += limit
        if offset >= total or frame.empty:
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_jolpica_drivers(season: int | str = "current") -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        url = f"{JOLPICA_BASE_URL}/{season}/drivers.json?limit=100&offset={offset}"
        payload = _get_json(url)
        races = _ergast_table(payload)
        for race in races:
            for d in race.get("Drivers", []):
                rows.append({
                    "driver_id": d.get("driverId"),
                    "code": d.get("code"),
                    "given_name": d.get("givenName"),
                    "family_name": d.get("familyName"),
                    "date_of_birth": d.get("dateOfBirth"),
                    "nationality": d.get("nationality"),
                })
        total = int(payload.get("MRData", {}).get("total", len(rows)))
        offset += 100
        if offset >= total or not races:
            break
    return pd.DataFrame(rows).drop_duplicates("driver_id")


def fetch_jolpica_constructors(season: int | str = "current") -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        url = f"{JOLPICA_BASE_URL}/{season}/constructors.json?limit=100&offset={offset}"
        payload = _get_json(url)
        races = _ergast_table(payload)
        for race in races:
            for c in race.get("Constructors", []):
                rows.append({
                    "constructor_id": c.get("constructorId"),
                    "constructor_name": c.get("name"),
                    "nationality": c.get("nationality"),
                })
        total = int(payload.get("MRData", {}).get("total", len(rows)))
        offset += 100
        if offset >= total or not races:
            break
    return pd.DataFrame(rows).drop_duplicates("constructor_id")


def fetch_jolpica_schedule(season: int | str = "current") -> pd.DataFrame:
    url = f"{JOLPICA_BASE_URL}/{season}/races.json?limit=100"
    payload = _get_json(url)
    rows = []
    for race in _ergast_table(payload):
        rows.append({
            "season": int(race["season"]),
            "round": int(race["round"]),
            "race_name": race["raceName"],
            "circuit_id": race["Circuit"]["circuitId"],
            "circuit_name": race["Circuit"]["circuitName"],
            "date": race["date"],
            "time": race.get("time"),
        })
    return pd.DataFrame(rows)


def _read_csv_if_exists(data_dir: str, name: str) -> pd.DataFrame:
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fallback file not found: {path}")
    return pd.read_csv(path)


def validate_dataset_schema(
    results: pd.DataFrame,
    drivers: pd.DataFrame,
    constructors: pd.DataFrame,
) -> None:
    required = {
        "results": {"season", "round", "driver_id", "constructor_id", "grid", "position", "points"},
        "drivers": {"driver_id"},
        "constructors": {"constructor_id"},
    }
    for name, frame, cols in [
        ("results", results, required["results"]),
        ("drivers", drivers, required["drivers"]),
        ("constructors", constructors, required["constructors"]),
    ]:
        missing = cols - set(frame.columns)
        if missing:
            raise ValueError(f"{name}.csv missing required columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{name} is empty.")
    if results["driver_id"].isna().any():
        raise ValueError("results.driver_id contains null values.")
    if results["constructor_id"].isna().any():
        raise ValueError("results.constructor_id contains null values.")


def load_and_clean_data(
    season: int = 2026,
    data_dir: str = "data",
    use_api: bool = True,
) -> F1DataBundle:
    """
    Load 2026 data from Jolpica, with CSV fallback.

    Expected CSVs:
      data/results.csv
      data/drivers.csv
      data/constructors.csv
    """
    results = drivers = constructors = schedule = None

    if use_api:
        try:
            results = fetch_jolpica_results(season)
            drivers = fetch_jolpica_drivers(season)
            constructors = fetch_jolpica_constructors(season)
            schedule = fetch_jolpica_schedule(season)
            LOGGER.info("Loaded 2026 data from Jolpica.")
        except Exception as exc:
            LOGGER.warning("Jolpica ingestion failed; using CSV fallback: %s", exc)

    if results is None or results.empty:
        results = _read_csv_if_exists(data_dir, "results.csv")
    if drivers is None or drivers.empty:
        drivers = _read_csv_if_exists(data_dir, "drivers.csv")
    if constructors is None or constructors.empty:
        constructors = _read_csv_if_exists(data_dir, "constructors.csv")
    if schedule is None or schedule.empty:
        schedule_path = os.path.join(data_dir, "schedule.csv")
        schedule = pd.read_csv(schedule_path) if os.path.exists(schedule_path) else None

    results = results.copy()
    for col in ["season", "round", "grid", "position", "points", "laps", "fastest_lap_rank"]:
        if col in results:
            results[col] = pd.to_numeric(results[col], errors="coerce")
    results["date"] = pd.to_datetime(results.get("date"), errors="coerce")
    results = results.drop_duplicates(
        subset=["season", "round", "driver_id"], keep="last"
    )
    results = results.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)

    drivers = drivers.drop_duplicates("driver_id").reset_index(drop=True)
    constructors = constructors.drop_duplicates("constructor_id").reset_index(drop=True)

    validate_dataset_schema(results, drivers, constructors)
    return F1DataBundle(results, drivers, constructors, schedule)


def _status_failure(status: str | float | None) -> int:
    if not isinstance(status, str):
        return 0
    success_terms = {"finished", "lapped", "lap", "classified"}
    s = status.strip().lower()
    return int(not any(term in s for term in success_terms))


def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create leakage-aware race-level features.

    The target is finish position. Rolling features use shift(1), ensuring that
    the current race result never leaks into its own feature vector.
    """
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.sort_values(["season", "round", "driver_id"]).reset_index(drop=True)

    for c in ["grid", "position", "points", "laps"]:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    work["dnf_flag"] = work["status"].map(_status_failure).astype(float)
    work["finish_position_clean"] = work["position"].clip(lower=1, upper=30)

    g = work.groupby("driver_id", group_keys=False)
    work["driver_form_3"] = g["finish_position_clean"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    work["driver_form_5"] = g["finish_position_clean"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    work["driver_points_5"] = g["points"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    work["driver_dnf_rate_5"] = g["dnf_flag"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )

    cg = work.groupby("constructor_id", group_keys=False)
    work["constructor_form_5"] = cg["finish_position_clean"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    )
    work["constructor_dnf_rate_10"] = cg["dnf_flag"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    )

    work["grid_position_factor"] = 1.0 / work["grid"].clip(lower=1)
    work["grid_delta"] = work["position"] - work["grid"]

    # 2026 technical-regulation features. These are constant regime descriptors
    # and can later be replaced with measured team-level engineering features.
    for key, value in RULES_2026.items():
        work[key] = value

    # Optional circuit/date signal.
    work["round_progress"] = work["round"] / work.groupby("season")["round"].transform("max")
    work["is_sprint_weekend"] = 0.0

    feature_cols = [
        "grid",
        "driver_form_3",
        "driver_form_5",
        "driver_points_5",
        "driver_dnf_rate_5",
        "constructor_form_5",
        "constructor_dnf_rate_10",
        "grid_position_factor",
        "active_aero",
        "overtake_mode",
        "drs_legacy",
        "mgu_h_present",
        "electric_power_kw",
        "ice_power_kw",
        "electric_power_share",
        "sustainable_fuel",
        "minimum_car_mass_kg",
        "wheelbase_mm",
        "car_width_mm",
        "front_tyre_width_reduction_mm",
        "rear_tyre_width_reduction_mm",
        "downforce_reduction_pct",
        "drag_reduction_pct",
        "round_progress",
        "is_sprint_weekend",
    ]

    model_df = work.dropna(subset=["position"]).copy()
    X = model_df[feature_cols]
    y = model_df["position"].astype(float)
    return X, y


def train_f1_model(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """Train a robust RandomForest regression pipeline with median imputation."""
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=500,
                    max_depth=12,
                    min_samples_leaf=3,
                    max_features=0.8,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X, y)
    # Store metadata for downstream simulation.
    model.feature_columns_ = list(X.columns)
    model.training_rows_ = len(X)
    return model


def _race_points(position: int) -> int:
    return RACE_POINTS.get(int(position), 0)


def _normalised_strength(driver: Mapping, keys: Sequence[str]) -> float:
    vals = [float(driver.get(k, 0.0) or 0.0) for k in keys]
    return float(np.mean(vals)) if vals else 0.0


def _build_simulation_frame(
    drivers_list: Sequence,
    model,
    remaining_race_index: int,
    weather_factor: float,
    reliability_weight: float,
) -> pd.DataFrame:
    """
    Convert driver dictionaries into the model's feature schema.

    A driver item can be a name string or a mapping. Mapping fields commonly used:
    name, grid, driver_form_3, driver_form_5, driver_points_5,
    driver_dnf_rate_5, constructor_form_5, constructor_dnf_rate_10.
    """
    rows = []
    feature_cols = getattr(model, "feature_columns_", None)
    if feature_cols is None:
        raise ValueError("Model has no feature_columns_; train it with train_f1_model().")

    for idx, item in enumerate(drivers_list):
        if isinstance(item, str):
            d = {"name": item}
        else:
            d = dict(item)
        row = {c: d.get(c, 0.0) for c in feature_cols}
        row["grid"] = d.get("grid", d.get("current_grid", 10 + idx % 10))
        row["driver_form_3"] = d.get("driver_form_3", 10.0)
        row["driver_form_5"] = d.get("driver_form_5", row["driver_form_3"])
        row["driver_points_5"] = d.get("driver_points_5", 8.0)
        row["driver_dnf_rate_5"] = d.get("driver_dnf_rate_5", 0.10)
        row["constructor_form_5"] = d.get("constructor_form_5", 10.0)
        row["constructor_dnf_rate_10"] = d.get("constructor_dnf_rate_10", 0.10)
        row["grid_position_factor"] = 1.0 / max(float(row["grid"]), 1.0)
        row["round_progress"] = d.get("round_progress", 0.75 + 0.01 * remaining_race_index)
        for key, value in RULES_2026.items():
            if key in feature_cols:
                row[key] = d.get(key, value)
        row["is_sprint_weekend"] = d.get("is_sprint_weekend", 0.0)
        rows.append(row)
    return pd.DataFrame(rows, index=[d.get("name", d.get("driver_id", str(i))) if isinstance(d, dict) else d
                                    for i, d in enumerate(drivers_list)])[feature_cols]


def run_monte_carlo_simulation(
    model,
    drivers_list: Sequence,
    num_simulations: int = 5000,
    weather_factor: float = 0.10,
    reliability_weight: float = 1.0,
    remaining_races: int = 0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run championship simulations.

    If remaining_races is 0, the function infers one future race. In production,
    pass the exact count of races remaining from the schedule.

    Returns a DataFrame with:
      driver, win_probability, podium_probability, expected_points,
      championship_position_mean, championship_position_std
    """
    if num_simulations < 1:
        raise ValueError("num_simulations must be >= 1.")
    if not drivers_list:
        raise ValueError("drivers_list must not be empty.")

    rng = np.random.default_rng(random_state)
    n_drivers = len(drivers_list)
    n_races = max(int(remaining_races), 1)

    totals = np.zeros((num_simulations, n_drivers), dtype=float)

    for race_idx in range(n_races):
        X = _build_simulation_frame(
            drivers_list, model, race_idx, weather_factor, reliability_weight
        )
        base_pred = np.asarray(model.predict(X), dtype=float)

        # Lower predicted finishing position is better. Reliability is represented
        # as an additional expected-position penalty; weather expands variance.
        dnfs = np.array([
            float((d.get("driver_dnf_rate_5", 0.10) if isinstance(d, dict) else 0.10) or 0.10)
            for d in drivers_list
        ])
        reliability_penalty = reliability_weight * dnfs * 4.0
        means = base_pred + reliability_penalty

        # Weather increases uncertainty; reliability also increases tail risk.
        sigma = 1.25 + (2.75 * float(np.clip(weather_factor, 0.0, 1.0))) + dnfs * 2.0

        sampled = rng.normal(
            loc=means[None, :],
            scale=sigma[None, :],
            size=(num_simulations, n_drivers),
        )

        # Race outcome is a ranking, not independent finishing positions.
        sampled = np.clip(sampled, 1.0, float(n_drivers) + 2.0)
        order = np.argsort(sampled, axis=1)
        positions = np.empty_like(order)
        row_idx = np.arange(num_simulations)[:, None]
        positions[row_idx, order] = np.arange(1, n_drivers + 1)[None, :]

        # Small DNF lottery after ranking to capture catastrophic reliability.
        dnf_draw = rng.random((num_simulations, n_drivers))
        dnf_mask = dnf_draw < np.clip(dnfs[None, :] * (1.0 + weather_factor), 0, 0.35)
        positions[dnf_mask] = n_drivers + 1

        for j in range(n_drivers):
            totals[:, j] += np.vectorize(_race_points)(positions[:, j])

    names = [
        (d.get("name") or d.get("driver_name") or d.get("driver_id") or f"Driver {i+1}")
        if isinstance(d, dict) else str(d)
        for i, d in enumerate(drivers_list)
    ]

    final_order = np.argsort(-totals, axis=1)
    championship_positions = np.empty_like(final_order)
    championship_positions[np.arange(num_simulations)[:, None], final_order] = (
        np.arange(1, n_drivers + 1)[None, :]
    )

    rows = []
    for j, name in enumerate(names):
        pos = championship_positions[:, j]
        rows.append({
            "driver": name,
            "win_probability": float(np.mean(pos == 1)),
            "podium_probability": float(np.mean(pos <= 3)),
            "expected_points": float(np.mean(totals[:, j])),
            "championship_position_mean": float(np.mean(pos)),
            "championship_position_std": float(np.std(pos)),
        })

    return pd.DataFrame(rows).sort_values(
        ["win_probability", "expected_points"], ascending=False
    ).reset_index(drop=True)


def build_driver_profiles(bundle: F1DataBundle) -> list[dict]:
    """Create current-form profiles suitable for simulation."""
    r = bundle.results.copy().sort_values(["driver_id", "round"])
    profiles = []
    latest = r.groupby("driver_id").tail(1)
    for driver_id, g in r.groupby("driver_id"):
        row = latest[latest["driver_id"] == driver_id].iloc[0]
        constructor = row.get("constructor_id")
        cg = r[r["constructor_id"] == constructor].tail(10)
        profiles.append({
            "name": row.get("driver_name") or driver_id,
            "driver_id": driver_id,
            "grid": float(row.get("grid") if pd.notna(row.get("grid")) else 10),
            "driver_form_3": float(g["position"].tail(3).mean()),
            "driver_form_5": float(g["position"].tail(5).mean()),
            "driver_points_5": float(g["points"].tail(5).mean()),
            "driver_dnf_rate_5": float(g["status"].tail(5).map(_status_failure).mean()),
            "constructor_form_5": float(cg["position"].mean()) if not cg.empty else 10.0,
            "constructor_dnf_rate_10": float(cg["status"].map(_status_failure).mean())
            if not cg.empty else 0.10,
        })
    return profiles
