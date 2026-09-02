"""
Synthetic KShop Sales Dataset Generator.

This module generates a reproducible synthetic retail sales dataset
for the AI Sales Forecasting portfolio project.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

START_DATE = "2025-09-01"
END_DATE = "2026-08-31"
RANDOM_SEED = 42

OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "kshop_sales_synthetic.csv"


# ============================================================
# KShop product catalog
# ============================================================

PRODUCTS = [
    {
        "product_id": 1,
        "product_name": "Produit 001",
        "category": "Alimentaire",
        "unit_price": 1200,
        "base_demand": 2.0,
    },
    {
        "product_id": 2,
        "product_name": "Eau Vive 1.5L",
        "category": "Boissons",
        "unit_price": 1000,
        "base_demand": 8.0,
    },
    {
        "product_id": 3,
        "product_name": "Coca-Cola 33cl",
        "category": "Boissons",
        "unit_price": 1500,
        "base_demand": 10.0,
    },
    {
        "product_id": 4,
        "product_name": "Fanta Orange 33cl",
        "category": "Boissons",
        "unit_price": 1400,
        "base_demand": 7.0,
    },
    {
        "product_id": 5,
        "product_name": "Jus de fruit 1L",
        "category": "Boissons",
        "unit_price": 3000,
        "base_demand": 4.0,
    },
    {
        "product_id": 6,
        "product_name": "Riz 1kg",
        "category": "Produits alimentaires",
        "unit_price": 3500,
        "base_demand": 6.0,
    },
    {
        "product_id": 7,
        "product_name": "Sucre 1kg",
        "category": "Produits alimentaires",
        "unit_price": 3200,
        "base_demand": 5.0,
    },
    {
        "product_id": 8,
        "product_name": "Huile alimentaire 1L",
        "category": "Produits alimentaires",
        "unit_price": 5500,
        "base_demand": 4.0,
    },
    {
        "product_id": 9,
        "product_name": "Lait en poudre 400g",
        "category": "Produits alimentaires",
        "unit_price": 9000,
        "base_demand": 2.5,
    },
    {
        "product_id": 10,
        "product_name": "Biscuits Chocolat",
        "category": "Snacks",
        "unit_price": 1800,
        "base_demand": 6.0,
    },
    {
        "product_id": 11,
        "product_name": "Chips Nature",
        "category": "Snacks",
        "unit_price": 1500,
        "base_demand": 5.0,
    },
    {
        "product_id": 12,
        "product_name": "Savon de toilette",
        "category": "Hygiène",
        "unit_price": 1300,
        "base_demand": 4.0,
    },
    {
        "product_id": 13,
        "product_name": "Dentifrice 75ml",
        "category": "Hygiène",
        "unit_price": 3000,
        "base_demand": 3.0,
    },
    {
        "product_id": 14,
        "product_name": "Liquide vaisselle 500ml",
        "category": "Produits ménagers",
        "unit_price": 2500,
        "base_demand": 3.0,
    },
]


# ============================================================
# Demand factors
# ============================================================

WEEKDAY_FACTORS = {
    0: 0.90,  # Monday
    1: 0.95,  # Tuesday
    2: 1.00,  # Wednesday
    3: 1.05,  # Thursday
    4: 1.20,  # Friday
    5: 1.30,  # Saturday
    6: 0.75,  # Sunday
}


MONTH_FACTORS = {
    1: 0.85,
    2: 0.90,
    3: 0.95,
    4: 1.00,
    5: 1.00,
    6: 1.05,
    7: 1.05,
    8: 1.10,
    9: 1.00,
    10: 1.05,
    11: 1.15,
    12: 1.35,
}


# ============================================================
# Helper functions
# ============================================================

def calculate_demand(
    base_demand: float,
    date: pd.Timestamp,
    rng: np.random.Generator,
) -> int:
    """
    Calculate daily product demand using business factors
    and controlled random variation.
    """

    weekday_factor = WEEKDAY_FACTORS[date.weekday()]
    month_factor = MONTH_FACTORS[date.month]

    # Long-term trend during the year.
    days_from_start = (date - pd.Timestamp(START_DATE)).days
    trend_factor = 1.0 + (days_from_start / 365) * 0.10

    expected_demand = (
        base_demand
        * weekday_factor
        * month_factor
        * trend_factor
    )

    # Controlled random variation.
    noise_factor = rng.normal(loc=1.0, scale=0.15)

    expected_demand *= max(noise_factor, 0.1)

    # Poisson distribution creates realistic integer demand.
    quantity = rng.poisson(max(expected_demand, 0))

    return int(quantity)


def generate_dataset() -> pd.DataFrame:
    """Generate the complete synthetic sales dataset."""

    rng = np.random.default_rng(RANDOM_SEED)

    dates = pd.date_range(
        start=START_DATE,
        end=END_DATE,
        freq="D",
    )

    rows = []

    for date in dates:
        for product in PRODUCTS:
            quantity = calculate_demand(
                base_demand=product["base_demand"],
                date=date,
                rng=rng,
            )

            revenue = quantity * product["unit_price"]

            rows.append(
                {
                    "date": date.date(),
                    "product_id": product["product_id"],
                    "product_name": product["product_name"],
                    "category": product["category"],
                    "quantity": quantity,
                    "unit_price": product["unit_price"],
                    "revenue": revenue,
                }
            )

    return pd.DataFrame(rows)


def validate_dataset(df: pd.DataFrame) -> None:
    """Run basic data-quality checks."""

    required_columns = {
        "date",
        "product_id",
        "product_name",
        "category",
        "quantity",
        "unit_price",
        "revenue",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    if df["quantity"].isna().any():
        raise ValueError("Quantity contains missing values.")

    if (df["quantity"] < 0).any():
        raise ValueError("Quantity contains negative values.")

    if (df["unit_price"] <= 0).any():
        raise ValueError("Invalid unit price detected.")

    expected_revenue = df["quantity"] * df["unit_price"]

    if not np.array_equal(
        df["revenue"].to_numpy(),
        expected_revenue.to_numpy(),
    ):
        raise ValueError("Revenue calculation is inconsistent.")

    if df["date"].min().isoformat() != START_DATE:
        raise ValueError("Unexpected start date.")

    if df["date"].max().isoformat() != END_DATE:
        raise ValueError("Unexpected end date.")


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Generate, validate and save the synthetic dataset."""

    print("AI Sales Forecasting")
    print("Synthetic dataset generator")
    print("-" * 50)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = generate_dataset()

    validate_dataset(df)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print(f"Dataset generated successfully.")
    print(f"Rows      : {len(df):,}")
    print(f"Products  : {df['product_id'].nunique()}")
    print(f"Start date: {df['date'].min()}")
    print(f"End date  : {df['date'].max()}")
    print(f"Output    : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()