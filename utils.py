"""
utils.py — SegmentIQ Shared Utilities
======================================
• Currency registry & formatting
• Smart column auto-detection (fuzzy alias matching + heuristic fallback)
• Robust data cleaning — NEVER blocks processing due to missing columns/rows
• RFM feature engineering
• Cluster label interpretation
"""

import re
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional

# ─────────────────────────────────────────────────────────────
# Currency Registry
# ─────────────────────────────────────────────────────────────

CURRENCIES: Dict[str, Dict] = {
    "USD": {"code": "USD", "symbol": "$",   "name": "US Dollar"},
    "EUR": {"code": "EUR", "symbol": "€",   "name": "Euro"},
    "GBP": {"code": "GBP", "symbol": "£",   "name": "British Pound"},
    "INR": {"code": "INR", "symbol": "₹",   "name": "Indian Rupee"},
    "NPR": {"code": "NPR", "symbol": "रू",  "name": "Nepalese Rupee"},
    "JPY": {"code": "JPY", "symbol": "¥",   "name": "Japanese Yen"},
    "CNY": {"code": "CNY", "symbol": "¥",   "name": "Chinese Yuan"},
    "AUD": {"code": "AUD", "symbol": "A$",  "name": "Australian Dollar"},
    "CAD": {"code": "CAD", "symbol": "C$",  "name": "Canadian Dollar"},
    "CHF": {"code": "CHF", "symbol": "Fr",  "name": "Swiss Franc"},
    "SGD": {"code": "SGD", "symbol": "S$",  "name": "Singapore Dollar"},
    "AED": {"code": "AED", "symbol": "د.إ", "name": "UAE Dirham"},
    "BRL": {"code": "BRL", "symbol": "R$",  "name": "Brazilian Real"},
    "MXN": {"code": "MXN", "symbol": "MX$", "name": "Mexican Peso"},
    "KRW": {"code": "KRW", "symbol": "₩",   "name": "South Korean Won"},
    "ZAR": {"code": "ZAR", "symbol": "R",   "name": "South African Rand"},
    "PKR": {"code": "PKR", "symbol": "₨",   "name": "Pakistani Rupee"},
    "BDT": {"code": "BDT", "symbol": "৳",   "name": "Bangladeshi Taka"},
    "THB": {"code": "THB", "symbol": "฿",   "name": "Thai Baht"},
    "IDR": {"code": "IDR", "symbol": "Rp",  "name": "Indonesian Rupiah"},
}


def format_currency(value: float, symbol: str) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return f"{symbol}0"
    if abs(value) >= 1_000_000:
        return f"{symbol}{value / 1_000_000:,.2f}M"
    if abs(value) >= 1_000:
        return f"{symbol}{value / 1_000:,.1f}K"
    return f"{symbol}{value:,.0f}"


# ─────────────────────────────────────────────────────────────
# Column Alias Map
# ─────────────────────────────────────────────────────────────

_COL_ALIASES: Dict[str, List[str]] = {
    "customer_id": [
        "customer_id", "customerid", "customer_id", "cust_id", "custid",
        "client_id", "clientid", "user_id", "userid", "member_id", "memberid",
        "buyer_id", "account_id", "contact_id", "customer", "client", "user",
    ],
    "invoice_id": [
        "invoice_id", "invoiceid", "invoice_no", "invoiceno", "invoice",
        "order_id", "orderid", "order_no", "orderno", "transaction_id", "txn_id",
        "receipt_id", "bill_id", "sale_id", "ref_id", "reference_id",
        "purchase_id", "order", "transaction",
    ],
    "invoice_date": [
        "invoice_date", "invoicedate", "order_date", "orderdate",
        "transaction_date", "txn_date", "purchase_date", "sale_date",
        "date", "created_at", "timestamp", "time", "datetime",
        "purchase_datetime", "order_datetime", "sale_datetime",
    ],
    "quantity": [
        "quantity", "qty", "units", "no_of_items", "item_count",
        "num_items", "pieces", "items", "unit_qty", "ordered_qty",
        "count", "volume",
    ],
    "unit_price": [
        "unit_price", "unitprice", "price", "price_per_unit", "unit_cost",
        "rate", "cost", "selling_price", "item_price", "product_price",
        "sale_price", "retail_price", "amount_per_unit", "value",
    ],
}


def _slug(s: str) -> str:
    """Lowercase, strip spaces/underscores for fuzzy match."""
    return re.sub(r"[\s_\-]+", "", s.lower().strip())


def _detect_columns(df: pd.DataFrame) -> Tuple[Dict[str, Optional[str]], List[str]]:
    """
    For each canonical column, find the best matching actual column.
    Uses exact alias match first, then slug-level fuzzy match.
    Returns (canonical→actual_or_None, list_of_still_missing).
    """
    raw_cols = list(df.columns)
    lower_map = {c.strip().lower().replace(" ", "_"): c for c in raw_cols}
    slug_map  = {_slug(c): c for c in raw_cols}

    detected: Dict[str, Optional[str]] = {}
    used: set = set()

    for canonical, aliases in _COL_ALIASES.items():
        found = None
        # 1. exact normalised match
        for alias in aliases:
            norm = alias.strip().lower().replace(" ", "_")
            if norm in lower_map and lower_map[norm] not in used:
                found = lower_map[norm]
                break
        # 2. slug fuzzy match
        if not found:
            for alias in aliases:
                sl = _slug(alias)
                if sl in slug_map and slug_map[sl] not in used:
                    found = slug_map[sl]
                    break
        # 3. keyword-in-column-name match
        if not found:
            keywords = {
                "customer_id": ["customer", "client", "user", "buyer", "member"],
                "invoice_id":  ["invoice", "order", "transaction", "receipt", "bill", "sale"],
                "invoice_date":["date", "time", "when", "purchase"],
                "quantity":    ["qty", "quant", "unit", "piece", "item", "count", "vol"],
                "unit_price":  ["price", "cost", "rate", "amount", "value", "selling"],
            }.get(canonical, [])
            for col in raw_cols:
                if col in used:
                    continue
                col_l = col.lower()
                if any(kw in col_l for kw in keywords):
                    found = col
                    break
        if found:
            used.add(found)
        detected[canonical] = found

    missing = [c for c, v in detected.items() if v is None]
    return detected, missing


def validate_schema(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Always returns True — we detect what we can and synthesise the rest.
    Returns (True, info_message_about_what_was_inferred).
    """
    detected, missing = _detect_columns(df)
    if not missing:
        return True, ""
    msgs = []
    for c in missing:
        msgs.append(f"'{c}' not found — will be synthesised automatically")
    return True, "; ".join(msgs)


# ─────────────────────────────────────────────────────────────
# Robust Data Cleaning — NEVER raises, always produces usable data
# ─────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Intelligently clean and impute the dataframe regardless of
    column names, missing values, or scrambled schema.
    Returns (cleaned_df, report_dict).
    """
    df = df.copy()
    detected, _ = _detect_columns(df)
    report: Dict = {"original_rows": len(df), "imputed": {}, "dropped": {}, "inferred": {}}

    # ── Rename detected columns to canonical names ──
    rename_map = {v: k for k, v in detected.items() if v is not None and v != k}
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
        report["inferred"] = {k: v for k, v in detected.items() if v and v != k}

    # ── customer_id ──
    if "customer_id" in df.columns:
        df["customer_id"] = df["customer_id"].astype(str).str.strip()
        df["customer_id"].replace(["nan", "none", "na", "", "null", "NaN"],
                                   np.nan, inplace=True)
        missing_mask = df["customer_id"].isna()
        if missing_mask.any():
            df.loc[missing_mask, "customer_id"] = "CUST_" + df.index[missing_mask].astype(str)
            report["imputed"]["customer_id"] = int(missing_mask.sum())
    else:
        df["customer_id"] = "CUST_" + df.index.astype(str)
        report["imputed"]["customer_id"] = len(df)

    # ── invoice_date ──
    if "invoice_date" in df.columns:
        df["invoice_date"] = pd.to_datetime(df["invoice_date"], errors="coerce")
        miss_d = df["invoice_date"].isna().sum()
        if miss_d > 0:
            valid = df["invoice_date"].dropna()
            fallback = valid.sort_values().iloc[len(valid) // 2] if len(valid) > 0 else pd.Timestamp("2023-06-01")
            df["invoice_date"] = df["invoice_date"].fillna(fallback)
            report["imputed"]["invoice_date"] = int(miss_d)
    else:
        df["invoice_date"] = pd.Timestamp("2023-06-01")
        report["imputed"]["invoice_date"] = len(df)

    # ── invoice_id ──
    if "invoice_id" in df.columns:
        df["invoice_id"] = df["invoice_id"].astype(str).str.strip()
        df["invoice_id"].replace(["nan", "none", "na", "", "null"],
                                  np.nan, inplace=True)
        miss_i = df["invoice_id"].isna().sum()
        if miss_i > 0:
            df["invoice_id"] = df["invoice_id"].fillna("INV_" + df.index.astype(str))
            report["imputed"]["invoice_id"] = int(miss_i)
    else:
        # Generate invoice IDs from customer + date
        df["invoice_id"] = (df["customer_id"].astype(str) + "_" +
                            df["invoice_date"].dt.strftime("%Y%m%d"))
        report["imputed"]["invoice_id"] = len(df)

    # ── quantity ──
    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
        miss_q = df["quantity"].isna().sum()
        if miss_q > 0:
            gmed = df["quantity"].median()
            gmed = gmed if pd.notna(gmed) else 1.0
            df["quantity"] = (
                df.groupby("customer_id")["quantity"]
                  .transform(lambda x: x.fillna(x.median() if x.notna().any() else gmed))
            )
            df["quantity"] = df["quantity"].fillna(gmed)
            report["imputed"]["quantity"] = int(miss_q)
        neg_q = (df["quantity"] <= 0).sum()
        if neg_q > 0:
            df = df[df["quantity"] > 0]
            report["dropped"]["negative_quantity"] = int(neg_q)
    else:
        df["quantity"] = 1.0
        report["imputed"]["quantity"] = len(df)

    # ── unit_price ──
    if "unit_price" in df.columns:
        df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
        miss_p = df["unit_price"].isna().sum()
        if miss_p > 0:
            gmed = df["unit_price"].median()
            gmed = gmed if pd.notna(gmed) else 1.0
            df["unit_price"] = (
                df.groupby("customer_id")["unit_price"]
                  .transform(lambda x: x.fillna(x.median() if x.notna().any() else gmed))
            )
            df["unit_price"] = df["unit_price"].fillna(gmed)
            report["imputed"]["unit_price"] = int(miss_p)
        neg_p = (df["unit_price"] <= 0).sum()
        if neg_p > 0:
            df = df[df["unit_price"] > 0]
            report["dropped"]["zero_price"] = int(neg_p)
    else:
        df["unit_price"] = 1.0
        report["imputed"]["unit_price"] = len(df)

    # ── line_total ──
    df["line_total"] = df["quantity"] * df["unit_price"]

    report["clean_rows"] = len(df)
    return df.reset_index(drop=True), report


# ─────────────────────────────────────────────────────────────
# RFM Feature Engineering
# ─────────────────────────────────────────────────────────────

def compute_rfm(df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    if snapshot_date is None:
        snapshot_date = df["invoice_date"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("customer_id")
        .agg(
            Recency=("invoice_date", lambda x: max((snapshot_date - x.max()).days, 0)),
            Frequency=("invoice_id", "nunique"),
            Monetary=("line_total", "sum"),
        )
        .reset_index()
    )
    # Keep only valid rows
    rfm = rfm[(rfm["Recency"] >= 0) & (rfm["Frequency"] > 0) & (rfm["Monetary"] > 0)]
    return rfm.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# Segment Templates
# ─────────────────────────────────────────────────────────────

SEGMENT_TEMPLATES = [
    {"name": "Champions",           "emoji": "🏆", "color": "#22c55e",
     "description": "Bought recently, buy often, and spend the most.",
     "action": "Reward them. Activate them as brand ambassadors."},
    {"name": "Loyal Customers",     "emoji": "💛", "color": "#f59e0b",
     "description": "Buy regularly with good frequency and consistent spend.",
     "action": "Offer loyalty programmes and early access to new products."},
    {"name": "At-Risk Customers",   "emoji": "⚠️", "color": "#f97316",
     "description": "Were good customers but haven't purchased recently.",
     "action": "Send personalised win-back campaigns with special offers."},
    {"name": "Lost / Inactive",     "emoji": "❌", "color": "#ef4444",
     "description": "Last purchase was a long time ago with low engagement.",
     "action": "Try reactivation offers or sunset them from active lists."},
    {"name": "New Customers",       "emoji": "🌱", "color": "#06b6d4",
     "description": "Recently joined with limited purchase history.",
     "action": "Onboard and nurture with a curated welcome series."},
    {"name": "Potential Loyalists", "emoji": "🔮", "color": "#8b5cf6",
     "description": "Recent buyers with growing frequency — loyalty signals emerging.",
     "action": "Offer membership tiers or personalised upsell opportunities."},
]


def assign_segment_labels(rfm_clustered: pd.DataFrame, n_clusters: int) -> Dict[int, Dict]:
    cluster_means = (
        rfm_clustered.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    )
    cluster_means["Score"] = (
        -cluster_means["Recency"].rank()
        + cluster_means["Frequency"].rank()
        + cluster_means["Monetary"].rank()
    )
    sorted_clusters = cluster_means["Score"].sort_values(ascending=False).index.tolist()
    templates = SEGMENT_TEMPLATES[:n_clusters]
    return {
        int(cid): templates[rank % len(templates)]
        for rank, cid in enumerate(sorted_clusters)
    }


def interpret_cluster(cluster_id: int, mapping: Dict[int, Dict]) -> Dict:
    return mapping.get(cluster_id, {
        "name": f"Segment {cluster_id}", "emoji": "📦", "color": "#94a3b8",
        "description": "No description available.", "action": "Analyse further.",
    })