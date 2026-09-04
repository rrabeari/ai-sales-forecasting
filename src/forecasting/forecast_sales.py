"""
AI Sales Forecasting
J7.3 - Recursive J+1 -> J+7 forecasting.

The forecasting process:
    Historical data
          ↓
        J+1
          ↓
    prediction added to working history
          ↓
        J+2
          ↓
        ...
          ↓
        J+7

Important:
- No future real quantity is used.
- The final trained model is used.
- Forecast quantities are clipped at zero because demand cannot be negative.
- Historical source files are never modified.
"""

from pathlib import Path

import joblib
import pandas as pd


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ML_READY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sales_ml_ready.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "final_model.joblib"
)

FORECAST_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "forecast"
)

FORECAST_FILE = (
    FORECAST_DIR
    / "sales_forecast_j1_j7.csv"
)


# ============================================================================
# FORECAST CONFIGURATION
# ============================================================================

FORECAST_HORIZON = 7

TARGET_COLUMN = "quantity"

FEATURE_COLUMNS = [
    "product_id",
    "day_of_week",
    "day_of_month",
    "month",
    "week_of_year",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
]


# ============================================================================
# SOURCE DATA VALIDATION
# ============================================================================

def validate_source_data(df: pd.DataFrame) -> None:
    """Validate the ML-ready historical dataset."""

    required_columns = [
        "date",
        "product_id",
        "product_name",
        "category",
        "quantity",
        "unit_price",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colonnes manquantes : {missing_columns}"
        )

    if df.empty:
        raise ValueError(
            "Le dataset historique est vide."
        )

    if df["date"].isna().any():
        raise ValueError(
            "Des dates NULL sont présentes."
        )

    if df["product_id"].isna().any():
        raise ValueError(
            "Des product_id NULL sont présents."
        )

    if df["quantity"].isna().any():
        raise ValueError(
            "Des quantités NULL sont présentes."
        )

    if (df["quantity"] < 0).any():
        raise ValueError(
            "Des quantités historiques négatives sont présentes."
        )

    duplicate_count = df.duplicated(
        subset=["date", "product_id"]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"{duplicate_count} doublon(s) date + product_id détecté(s)."
        )


# ============================================================================
# CALENDAR FEATURES
# ============================================================================

def build_calendar_features(
    forecast_date: pd.Timestamp,
) -> dict:
    """Build calendar features for one forecast date."""

    return {
        "day_of_week": forecast_date.dayofweek,
        "day_of_month": forecast_date.day,
        "month": forecast_date.month,
        "week_of_year": int(
            forecast_date.isocalendar().week
        ),
        "is_weekend": int(
            forecast_date.dayofweek >= 5
        ),
    }


# ============================================================================
# PRODUCT HISTORY
# ============================================================================

def get_product_history(
    history: pd.DataFrame,
    product_id,
) -> pd.DataFrame:
    """Return sorted quantity history for one product."""

    product_history = (
        history[
            history["product_id"] == product_id
        ]
        .sort_values("date")
        .copy()
    )

    if product_history.empty:
        raise ValueError(
            f"Aucun historique pour le produit {product_id}."
        )

    if len(product_history) < 30:
        raise ValueError(
            f"Historique insuffisant pour {product_id}: "
            f"{len(product_history)} observations."
        )

    return product_history


# ============================================================================
# LAG + ROLLING FEATURES
# ============================================================================

def build_demand_features(
    history: pd.DataFrame,
    product_id,
) -> dict:
    """
    Build lag and rolling features using only the history
    available immediately before the forecast date.
    """

    product_history = get_product_history(
        history,
        product_id,
    )

    quantities = product_history["quantity"]

    return {
        "lag_1": float(
            quantities.iloc[-1]
        ),
        "lag_7": float(
            quantities.iloc[-7]
        ),
        "lag_14": float(
            quantities.iloc[-14]
        ),
        "rolling_mean_7": float(
            quantities.iloc[-7:].mean()
        ),
        "rolling_mean_14": float(
            quantities.iloc[-14:].mean()
        ),
        "rolling_mean_30": float(
            quantities.iloc[-30:].mean()
        ),
    }


# ============================================================================
# BUILD ONE FORECAST INPUT
# ============================================================================

def build_forecast_input(
    history: pd.DataFrame,
    product_id,
    forecast_date: pd.Timestamp,
) -> pd.DataFrame:
    """Build one model input row for one product/date."""

    product_history = get_product_history(
        history,
        product_id,
    )

    latest_product_row = (
        product_history
        .sort_values("date")
        .iloc[-1]
    )

    calendar_features = build_calendar_features(
        forecast_date
    )

    demand_features = build_demand_features(
        history,
        product_id,
    )

    row = {
        "product_id": product_id,
        **calendar_features,
        **demand_features,
    }

    model_input = pd.DataFrame(
        [row],
        columns=FEATURE_COLUMNS,
    )

    return model_input


# ============================================================================
# PRODUCT METADATA
# ============================================================================

def build_product_metadata(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Extract one metadata row per product."""

    metadata = (
        df[
            [
                "product_id",
                "product_name",
                "category",
                "unit_price",
            ]
        ]
        .drop_duplicates("product_id")
        .sort_values("product_id")
        .reset_index(drop=True)
    )

    if metadata["product_id"].duplicated().any():
        raise ValueError(
            "Plusieurs métadonnées existent pour un même product_id."
        )

    return metadata


# ============================================================================
# ONE RECURSIVE FORECAST DAY
# ============================================================================

def forecast_one_day(
    model,
    history: pd.DataFrame,
    metadata: pd.DataFrame,
    forecast_date: pd.Timestamp,
    forecast_day: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Forecast one future day for every product.

    Returns:
        daily_forecast:
            Forecast results for the day.

        updated_history:
            Working history including the new predictions.
    """

    forecast_rows = []

    updated_history = history.copy()

    for _, product in metadata.iterrows():

        product_id = product["product_id"]

        model_input = build_forecast_input(
            history=updated_history,
            product_id=product_id,
            forecast_date=forecast_date,
        )

        prediction = float(
            model.predict(model_input)[0]
        )

        # Demand cannot be negative.
        prediction = max(0.0, prediction)

        unit_price = float(
            product["unit_price"]
        )

        forecast_revenue = (
            prediction * unit_price
        )

        forecast_rows.append(
            {
                "date": forecast_date,
                "product_id": product_id,
                "product_name": product["product_name"],
                "category": product["category"],
                "forecast_quantity": prediction,
                "unit_price": unit_price,
                "forecast_revenue": forecast_revenue,
                "forecast_day": f"J+{forecast_day}",
            }
        )

        # ---------------------------------------------------------------
        # IMPORTANT:
        # Add the prediction to the working history.
        #
        # This prediction can therefore become lag_1, lag_7, etc.
        # for subsequent forecast dates.
        #
        # This is what makes the forecast recursive.
        # ---------------------------------------------------------------

        history_row = {
            "date": forecast_date,
            "product_id": product_id,
            "product_name": product["product_name"],
            "category": product["category"],
            "quantity": prediction,
            "unit_price": unit_price,
        }

        updated_history = pd.concat(
            [
                updated_history,
                pd.DataFrame([history_row]),
            ],
            ignore_index=True,
        )

    daily_forecast = pd.DataFrame(
        forecast_rows
    )

    return daily_forecast, updated_history


# ============================================================================
# COMPLETE J+1 -> J+7 FORECAST
# ============================================================================

def run_recursive_forecast() -> pd.DataFrame:
    """Generate the complete recursive J+1 -> J+7 forecast."""

    print("=" * 70)
    print("J7.3 — FORECAST RÉCURSIF J+1 → J+7")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Load historical data
    # ------------------------------------------------------------------------

    if not ML_READY_FILE.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {ML_READY_FILE}"
        )

    df = pd.read_csv(
        ML_READY_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = (
        df
        .sort_values(
            ["product_id", "date"]
        )
        .reset_index(drop=True)
    )

    print(
        f"[OK] Dataset chargé : "
        f"{len(df):,} lignes"
    )

    print(
        f"[INFO] Période historique : "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    print(
        f"[INFO] Produits : "
        f"{df['product_id'].nunique()}"
    )

    validate_source_data(df)

    print(
        "[PASS] Dataset historique validé"
    )

    # ------------------------------------------------------------------------
    # Load final model
    # ------------------------------------------------------------------------

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Modèle final introuvable : {MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    if not hasattr(model, "predict"):
        raise TypeError(
            "Le modèle final ne possède pas predict()."
        )

    print(
        f"[OK] Modèle chargé : "
        f"{MODEL_FILE.name}"
    )

    print(
        "[PASS] Interface predict() disponible"
    )

    # ------------------------------------------------------------------------
    # Product metadata
    # ------------------------------------------------------------------------

    metadata = build_product_metadata(
        df
    )

    product_count = len(metadata)

    print(
        f"[PASS] Métadonnées produits : "
        f"{product_count} produits"
    )

    # ------------------------------------------------------------------------
    # Forecast starting point
    # ------------------------------------------------------------------------

    last_historical_date = df["date"].max()

    print(
        f"[INFO] Dernière date historique : "
        f"{last_historical_date.date()}"
    )

    # Working history:
    # this dataframe is modified only in memory.
    working_history = df[
        [
            "date",
            "product_id",
            "product_name",
            "category",
            "quantity",
            "unit_price",
        ]
    ].copy()

    all_forecasts = []

    # ------------------------------------------------------------------------
    # Recursive forecast
    # ------------------------------------------------------------------------

    for forecast_day in range(
        1,
        FORECAST_HORIZON + 1,
    ):

        forecast_date = (
            last_historical_date
            + pd.Timedelta(days=forecast_day)
        )

        print(
            f"[FORECAST] J+{forecast_day} "
            f"→ {forecast_date.date()}"
        )

        daily_forecast, working_history = (
            forecast_one_day(
                model=model,
                history=working_history,
                metadata=metadata,
                forecast_date=forecast_date,
                forecast_day=forecast_day,
            )
        )

        expected_rows = product_count

        if len(daily_forecast) != expected_rows:
            raise ValueError(
                f"J+{forecast_day}: "
                f"{len(daily_forecast)} prévisions "
                f"au lieu de {expected_rows}."
            )

        all_forecasts.append(
            daily_forecast
        )

        print(
            f"[PASS] J+{forecast_day} : "
            f"{len(daily_forecast)} produits"
        )

    # ------------------------------------------------------------------------
    # Combine all forecast days
    # ------------------------------------------------------------------------

    forecast = pd.concat(
        all_forecasts,
        ignore_index=True,
    )

    expected_total = (
        product_count
        * FORECAST_HORIZON
    )

    if len(forecast) != expected_total:
        raise ValueError(
            f"Nombre total incorrect : "
            f"{len(forecast)} au lieu de "
            f"{expected_total}."
        )

    # ------------------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------------------

    if forecast["forecast_quantity"].isna().any():
        raise ValueError(
            "Des prévisions NULL sont présentes."
        )

    if (
        forecast["forecast_quantity"] < 0
    ).any():
        raise ValueError(
            "Des prévisions négatives sont présentes."
        )

    if forecast["forecast_revenue"].isna().any():
        raise ValueError(
            "Des revenus prévisionnels NULL sont présents."
        )

    duplicate_count = forecast.duplicated(
        subset=["date", "product_id"]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"{duplicate_count} doublon(s) "
            f"date + product_id dans le forecast."
        )

    expected_dates = pd.date_range(
        start=last_historical_date
        + pd.Timedelta(days=1),
        periods=FORECAST_HORIZON,
        freq="D",
    )

    actual_dates = (
        pd.DatetimeIndex(
            sorted(
                forecast["date"].unique()
            )
        )
    )

    if not actual_dates.equals(
        expected_dates
    ):
        raise ValueError(
            "Les dates du forecast ne correspondent "
            "pas à J+1 → J+7."
        )

    # ------------------------------------------------------------------------
    # Save forecast
    # ------------------------------------------------------------------------

    FORECAST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast.to_csv(
        FORECAST_FILE,
        index=False,
    )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    print("-" * 70)

    print(
        f"[OK] Prévisions générées : "
        f"{len(forecast)}"
    )

    print(
        f"[INFO] Produits : "
        f"{forecast['product_id'].nunique()}"
    )

    print(
        f"[INFO] Jours : "
        f"{forecast['date'].nunique()}"
    )

    print(
        f"[INFO] Quantité prévisionnelle totale : "
        f"{forecast['forecast_quantity'].sum():.2f}"
    )

    print(
        f"[INFO] CA prévisionnel total : "
        f"{forecast['forecast_revenue'].sum():,.2f} AR"
    )

    print(
        f"[INFO] Prévision moyenne / ligne : "
        f"{forecast['forecast_quantity'].mean():.4f}"
    )

    print(
        f"[OK] Fichier sauvegardé : "
        f"{FORECAST_FILE}"
    )

    print("-" * 70)

    print(
        "[PASS] Aucune quantité future réelle utilisée"
    )

    print(
        "[PASS] Forecast récursif J+1 → J+7"
    )

    print(
        "[PASS] Aucun doublon date + product_id"
    )

    print(
        "[PASS] Toutes les prévisions sont non négatives"
    )

    print("=" * 70)
    print("J7.3 — FORECAST RÉCURSIF : OK")
    print("=" * 70)

    return forecast


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_recursive_forecast()