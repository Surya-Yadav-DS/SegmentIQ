"""
generate_data.py
================
PURPOSE: This file is ONLY for creating a synthetic demo dataset when you do
not yet have a real CSV file.  It has NO effect on how the model processes
your actual data — when you upload a real CSV the model reads every single row
and counts every unique customer without any limit.

The n_customers and n_rows parameters below control the SIZE of the demo
dataset that is generated.  They do NOT cap or limit your real data in any way.

If you already have your own CSV in Data/online_retail.csv, you can skip
running this file entirely.

Run:
    python generate_data.py                    # default: all customers, no row cap
    python generate_data.py --customers 2000   # generate larger demo
"""

import os
import random
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "Data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "online_retail.csv")


def generate_online_retail_data(n_customers: int = 1000) -> pd.DataFrame:
    """
    Generates a synthetic transactional dataset with realistic RFM patterns.

    Parameters
    ----------
    n_customers : int
        Number of unique synthetic customers to generate.
        All transactions for ALL customers are written — no row cap.
        Default changed from 500 to 1000 for a richer demo.

    Returns
    -------
    pd.DataFrame  with columns: customer_id, invoice_id, invoice_date,
                                quantity, unit_price
    """
    start_date = datetime(2022, 1, 1)   # wider date window for more diversity
    end_date   = datetime(2024, 1, 1)

    customer_ids = [f"C{str(i).zfill(5)}" for i in range(1001, 1001 + n_customers)]

    # Each tuple: (proportion, recency_days_range, n_invoices_range, spend_per_item_range)
    segments = {
        "champions": (0.12, (5,   30),  (10, 60), (80,   600)),
        "loyal":     (0.18, (20,  90),  (5,  25), (30,   250)),
        "at_risk":   (0.22, (90,  200), (3,  12), (20,   120)),
        "lost":      (0.18, (200, 400), (1,   6), (10,    60)),
        "new":       (0.15, (1,   20),  (1,   3), (25,   180)),
        "potential": (0.15, (10,  70),  (2,  10), (18,   100)),
    }

    rows = []
    invoice_counter = 10000

    for cid in customer_ids:
        seg_name = random.choices(
            list(segments.keys()),
            weights=[v[0] for v in segments.values()]
        )[0]
        _, recency_range, freq_range, spend_range = segments[seg_name]

        days_since = random.randint(*recency_range)
        last_dt    = end_date - timedelta(days=days_since)
        n_invoices = random.randint(*freq_range)

        for _ in range(n_invoices):
            # Spread invoices randomly across the customer's history
            inv_date = last_dt - timedelta(days=random.randint(0, 600))
            inv_date = max(inv_date, start_date)
            inv_id   = f"INV{invoice_counter}"
            invoice_counter += 1

            n_items = random.randint(1, 8)
            for _ in range(n_items):
                unit_price = round(
                    random.uniform(spend_range[0] / n_items,
                                   spend_range[1] / n_items), 2
                )
                quantity = random.randint(1, 6)
                rows.append({
                    "customer_id":  cid,
                    "invoice_id":   inv_id,
                    "invoice_date": inv_date.strftime("%Y-%m-%d"),
                    "quantity":     quantity,
                    "unit_price":   unit_price,
                })

    # Shuffle rows (realistic — rows are not ordered by customer)
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    # NO row cap — every transaction for every customer is preserved
    return df


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic demo data")
    p.add_argument("--customers", type=int, default=1000,
                   help="Number of unique customers (default: 1000)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Generating demo data for {args.customers:,} customers (no row cap) ...")
    df = generate_online_retail_data(n_customers=args.customers)

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\n✅ Generated {len(df):,} rows, {df['customer_id'].nunique():,} unique customers.")
    print(f"   Saved to:  {OUTPUT_PATH}")
    print(f"\n   NOTE: This demo data is only used if you do NOT upload your own CSV.")
    print(f"         When you upload a real file, ALL its rows and customers are read.")
    print()
    print(df.head(5).to_string(index=False))
