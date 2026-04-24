"""
generate_data.py
================
Generates a synthetic Online Retail demo dataset and saves it to the
correct local path relative to this script.

Run from the project folder:
    python generate_data.py

Output:
    <project_folder>/Data/online_retail.csv
"""

import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ── Save next to this script, in a "Data" sub-folder ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "Data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "online_retail.csv")


def generate_online_retail_data(n_customers: int = 500, n_rows: int = 15000) -> pd.DataFrame:
    start_date = datetime(2023, 1, 1)
    end_date   = datetime(2024, 1, 1)

    customer_ids = [f"C{str(i).zfill(5)}" for i in range(1001, 1001 + n_customers)]

    segments = {
        "champions": (0.15, (5,  30),  (10, 50), (50,  500)),
        "loyal":     (0.20, (30, 90),  (5,  20), (30,  200)),
        "at_risk":   (0.20, (90, 180), (3,  10), (20,  100)),
        "lost":      (0.15, (180,365), (1,  5),  (10,  50)),
        "new":       (0.15, (1,  15),  (1,  3),  (20,  150)),
        "potential": (0.15, (10, 60),  (2,  8),  (15,  80)),
    }

    rows = []
    invoice_counter = 10000

    for cid in customer_ids:
        seg = random.choices(list(segments.keys()),
                             weights=[v[0] for v in segments.values()])[0]
        _, recency_range, freq_range, spend_range = segments[seg]

        days_since = random.randint(*recency_range)
        last_dt    = end_date - timedelta(days=days_since)
        n_invoices = random.randint(*freq_range)

        for _ in range(n_invoices):
            inv_date = last_dt - timedelta(days=random.randint(0, 200))
            inv_date = max(inv_date, start_date)
            inv_id   = f"INV{invoice_counter}"
            invoice_counter += 1

            n_items = random.randint(1, 6)
            for _ in range(n_items):
                unit_price = round(
                    random.uniform(spend_range[0] / n_items,
                                   spend_range[1] / n_items), 2
                )
                quantity = random.randint(1, 5)
                rows.append({
                    "customer_id":  cid,
                    "invoice_id":   inv_id,
                    "invoice_date": inv_date.strftime("%Y-%m-%d"),
                    "quantity":     quantity,
                    "unit_price":   unit_price,
                })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    if len(df) > n_rows:
        df = df.head(n_rows)
    return df


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)          # create Data/ if it doesn't exist
    df = generate_online_retail_data()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Generated {len(df):,} rows, {df['customer_id'].nunique():,} customers.")
    print(f"   Saved to: {OUTPUT_PATH}")
    print(df.head(3).to_string(index=False))