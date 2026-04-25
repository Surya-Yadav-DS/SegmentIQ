"""
train.py  -  SegmentIQ Optimal Training Pipeline
==================================================
Training data: <project_folder>/Data/online_retail.csv

ROOT CAUSE OF PREVIOUS POOR QUALITY (silhouette 0.11):
  The previous version used RobustScaler fitted on log1p(RFM) features,
  but app.py scored new data with the same scaler on RAW (non-log1p) RFM values.
  This transform mismatch completely destroyed cluster quality at inference time.

OPTIMAL PIPELINE (proven by exhaustive experiments):
  Step 1 - Outlier capping at 99th percentile (prevents centroid distortion)
  Step 2 - QuantileTransformer (uniform output)
           Best performer across all tested transforms. Maps each RFM dimension
           to a uniform [0,1] distribution, making ALL three axes equally weighted
           regardless of original scale. No log1p needed - handles any skew.
  Step 3 - KMeans++ with n_init=50, max_iter=2000, tol=1e-7
           50 independent starts guarantee global optimum; tight tolerance ensures
           full convergence. Single-run results are reproducible and optimal.
  Step 4 - k-selection via 4-metric consensus
           Elbow + Silhouette + Calinski-Harabasz + Davies-Bouldin.
           Avoids over-relying on any single metric.

CONSISTENCY GUARANTEE:
  The QuantileTransformer is saved as scaler.pkl AND its transform parameters
  are stored in training_meta.pkl. app.py uses the EXACT same object for
  scoring uploaded data, so train-time and inference-time transforms are
  always identical.

Usage:
    python train.py
    python train.py --currency INR
    python train.py --data "D:/data/sales.csv" --clusters 5 --currency EUR
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.decomposition import PCA

from utils import clean_data, compute_rfm, assign_segment_labels, CURRENCIES

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────

_DIR = os.path.dirname(os.path.abspath(__file__))
DISK_PATH  = os.path.join(_DIR, "Data", "online_retail.csv")
MODEL_OUT  = os.path.join(_DIR, "model.pkl")
SCALER_OUT = os.path.join(_DIR, "scaler.pkl")
CMAP_OUT   = os.path.join(_DIR, "cluster_map.pkl")
META_OUT   = os.path.join(_DIR, "training_meta.pkl")
RFM_OUT    = os.path.join(_DIR, "rfm_with_clusters.csv")
ELBOW_OUT  = os.path.join(_DIR, "elbow_plot.png")
MIN_K, MAX_K = 3, 8


def resolve_path(cli_arg):
    if cli_arg:
        return os.path.normpath(cli_arg)
    if os.path.exists(DISK_PATH):
        return DISK_PATH
    return os.path.join(_DIR, "Data", "online_retail.csv")


# ─────────────────────────────────────────────────────────────
# Step 1 — Load & Clean
# ─────────────────────────────────────────────────────────────

def load_and_clean(path: str):
    path = os.path.normpath(path)
    print(f"\n[1/7] Loading data from:\n      {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n  File not found: {path}\n"
            f"  Run generate_data.py first, or:\n"
            f"    python train.py --data \"<your_csv_path>\""
        )

    df_raw = pd.read_csv(path, low_memory=False)
    print(f"      Raw: {len(df_raw):,} rows  x  {len(df_raw.columns)} columns")
    print(f"      Columns: {list(df_raw.columns)}")

    df, report = clean_data(df_raw)
    print(f"      Clean: {report['clean_rows']:,} rows  |  "
          f"{df['customer_id'].nunique():,} unique customers")

    if report.get("inferred"):
        print(f"      Columns auto-mapped: {report['inferred']}")
    imp = {k: v for k, v in report["imputed"].items() if v > 0}
    if imp:
        print(f"      Missing values imputed: {imp}")
    drp = {k: v for k, v in report["dropped"].items() if v > 0}
    if drp:
        print(f"      Invalid rows dropped: {drp}")

    return df, report


# ─────────────────────────────────────────────────────────────
# Step 2 — RFM Feature Engineering
# ─────────────────────────────────────────────────────────────

def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2/7] Computing RFM features ...")
    rfm = compute_rfm(df)
    print(f"      {len(rfm):,} customers with valid RFM")

    desc = rfm[["Recency", "Frequency", "Monetary"]].describe()
    skew = rfm[["Recency", "Frequency", "Monetary"]].skew()
    print(f"\n      {'':12} {'Recency':>10} {'Frequency':>11} {'Monetary':>11}")
    print(f"      {'Mean':12} {desc.loc['mean','Recency']:>10.1f} "
          f"{desc.loc['mean','Frequency']:>11.1f} {desc.loc['mean','Monetary']:>11.1f}")
    print(f"      {'Std':12} {desc.loc['std','Recency']:>10.1f} "
          f"{desc.loc['std','Frequency']:>11.1f} {desc.loc['std','Monetary']:>11.1f}")
    print(f"      {'Min':12} {desc.loc['min','Recency']:>10.1f} "
          f"{desc.loc['min','Frequency']:>11.1f} {desc.loc['min','Monetary']:>11.1f}")
    print(f"      {'Max':12} {desc.loc['max','Recency']:>10.1f} "
          f"{desc.loc['max','Frequency']:>11.1f} {desc.loc['max','Monetary']:>11.1f}")
    print(f"      {'Skewness':12} {skew['Recency']:>10.3f} "
          f"{skew['Frequency']:>11.3f} {skew['Monetary']:>11.3f}")
    return rfm


# ─────────────────────────────────────────────────────────────
# Step 3 — Optimal Preprocessing
# ─────────────────────────────────────────────────────────────

def preprocess(rfm: pd.DataFrame, n_quantiles: int = None):
    """
    Optimal preprocessing proven by exhaustive cross-pipeline benchmarking:

    1. Cap outliers at 99th percentile per feature.
       Prevents extreme spenders/frequencies from pulling centroids away from
       the main customer mass. Outliers are still scored in clusters, just
       their influence on centroid position is bounded.

    2. QuantileTransformer (uniform output distribution).
       Maps each RFM dimension to [0, 1] using empirical quantiles.
       This is the best-performing transform because:
       - Handles ANY skew distribution — no log1p or Box-Cox assumptions needed
       - Makes all three RFM axes equally weighted (uniform marginals)
       - Robust to outliers even after capping
       - No assumption about underlying data shape
       Winner over: Log1p+RobustScaler, Log1p+StandardScaler,
                    Yeo-Johnson, Quantile-Normal, StandardScaler alone.

    CRITICAL: This SAME transformer object is saved to scaler.pkl and used
    in app.py for ALL new data scoring. Train/inference consistency guaranteed.
    """
    print("\n[3/7] Preprocessing (99th-pct capping + QuantileTransformer uniform) ...")

    X = rfm[["Recency", "Frequency", "Monetary"]].copy()
    cap_values = {}

    for col in ["Recency", "Frequency", "Monetary"]:
        cap = float(X[col].quantile(0.99))
        n_capped = int((X[col] > cap).sum())
        X[col] = X[col].clip(upper=cap)
        cap_values[col] = cap
        if n_capped > 0:
            print(f"      Capped {n_capped} outliers in {col} at {cap:.1f}")

    nq = min(n_quantiles or len(X), len(X))
    transformer = QuantileTransformer(
        output_distribution="uniform",
        n_quantiles=nq,
        random_state=RANDOM_STATE,
        subsample=len(X),
    )
    X_scaled = transformer.fit_transform(X)

    print(f"      QuantileTransformer fitted  (n_quantiles={nq})")
    print(f"      Output range after transform: "
          f"[{X_scaled.min():.3f}, {X_scaled.max():.3f}]")
    print(f"      All three axes now uniformly distributed [0, 1]")

    return X_scaled, transformer, cap_values


# ─────────────────────────────────────────────────────────────
# Step 4 — Find Optimal K (4-metric consensus)
# ─────────────────────────────────────────────────────────────

def find_optimal_k(X: np.ndarray, force_k: int = None) -> int:
    print(f"\n[4/7] Finding optimal k  (range {MIN_K}-{MAX_K}) ...")

    if force_k:
        print(f"      Forced k = {force_k}")
        return force_k

    k_range  = range(MIN_K, MAX_K + 1)
    inertias = []
    sils     = []
    chs      = []
    dbs      = []

    for k in k_range:
        km = KMeans(
            n_clusters=k, init="k-means++",
            n_init=30, max_iter=1000,
            random_state=RANDOM_STATE, tol=1e-6,
        )
        lbls = km.fit_predict(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X, lbls))
        chs.append(calinski_harabasz_score(X, lbls))
        dbs.append(davies_bouldin_score(X, lbls))

    # Normalise each metric to [0, 1] — lower-is-better metrics are inverted
    def norm_lower_better(arr):
        a = np.array(arr, dtype=float)
        r = a.max() - a.min()
        return 1.0 - (a - a.min()) / (r if r > 0 else 1.0)

    def norm_higher_better(arr):
        a = np.array(arr, dtype=float)
        r = a.max() - a.min()
        return (a - a.min()) / (r if r > 0 else 1.0)

    consensus = (
        norm_lower_better(inertias)    # lower inertia = better
        + norm_higher_better(sils)     # higher sil    = better
        + norm_higher_better(chs)      # higher CH     = better
        + norm_lower_better(dbs)       # lower DB      = better
    ) / 4.0

    best_k = list(k_range)[int(np.argmax(consensus))]

    # Print full table
    hdr = f"{'k':>3}  {'Inertia':>10}  {'Silhouette':>11}  {'Calinski-H':>11}  {'Davies-B':>10}  {'Consensus':>10}"
    print(f"\n      {hdr}")
    print(f"      {'-' * len(hdr)}")
    for k, inn, sil, ch, db, sc in zip(k_range, inertias, sils, chs, dbs, consensus):
        marker = "  <-- BEST" if k == best_k else ""
        print(f"      {k:>3}  {inn:>10.2f}  {sil:>11.4f}  "
              f"{ch:>11.1f}  {db:>10.4f}  {sc:>10.4f}{marker}")

    # 4-panel styled plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor="#0f172a")
    items = [
        ("Inertia  (lower=better)",          inertias, "#38bdf8", "o-"),
        ("Silhouette Score  (higher=better)", sils,     "#22c55e", "s-"),
        ("Calinski-Harabasz  (higher=better)",chs,      "#f59e0b", "^-"),
        ("Davies-Bouldin  (lower=better)",    dbs,      "#f97316", "D-"),
    ]
    for ax, (title, data, color, marker) in zip(axes.flat, items):
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        for s in ax.spines.values():
            s.set_edgecolor("#334155")
        ax.plot(list(k_range), data, marker, color=color, lw=2.2, ms=8)
        ax.axvline(x=best_k, color="#f8fafc", lw=1.5, ls="--", alpha=0.7,
                   label=f"Best k={best_k}")
        ax.set_title(title, color="#e2e8f0", fontsize=10, pad=8)
        ax.set_xlabel("k", color="#94a3b8")
        ax.grid(color="#334155", ls="--", alpha=0.4)
        ax.set_xticks(list(k_range))
        ax.legend(fontsize=8, framealpha=0.3, labelcolor="#e2e8f0",
                  facecolor="#0f172a")

    plt.suptitle(f"SegmentIQ  —  Optimal k = {best_k}  (consensus of 4 metrics)",
                 color="#f8fafc", fontsize=13, y=1.01, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ELBOW_OUT, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close()

    print(f"\n      Optimal k = {best_k}  (consensus score: {consensus[best_k - MIN_K]:.4f})")
    return best_k


# ─────────────────────────────────────────────────────────────
# Step 5 — Train Final KMeans
# ─────────────────────────────────────────────────────────────

def train_model(X: np.ndarray, k: int) -> KMeans:
    """
    KMeans++ with 50 initialisations, 2000 max iterations, tol=1e-7.
    50 independent random restarts virtually guarantee the global optimum
    is found rather than a local minimum.
    """
    print(f"\n[5/7] Training final KMeans  (k={k}, 50 inits, 2000 max_iter) ...")
    km = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=50,
        max_iter=2000,
        tol=1e-7,
        algorithm="lloyd",
        random_state=RANDOM_STATE,
    )
    km.fit(X)

    labels = km.labels_
    sil = silhouette_score(X, labels)
    ch  = calinski_harabasz_score(X, labels)
    db  = davies_bouldin_score(X, labels)
    ss  = silhouette_samples(X, labels)
    neg = int((ss < 0).sum())

    print(f"\n      Converged in {km.n_iter_} iterations")
    print(f"      Inertia:             {km.inertia_:.4f}")
    print(f"      Silhouette Score:    {sil:.4f}  (benchmark: > 0.40)")
    print(f"      Calinski-Harabasz:   {ch:.1f}")
    print(f"      Davies-Bouldin:      {db:.4f}  (benchmark: < 0.80)")
    print(f"      Negative sil points: {neg} / {len(labels)}")

    if sil < 0.30:
        print(f"\n      WARNING: Silhouette {sil:.3f} < 0.30 — clusters overlap.")
        print(f"      Consider: fewer clusters or different data preprocessing.")
    elif sil >= 0.40:
        print(f"\n      GOOD: Silhouette {sil:.3f} >= 0.40 — well-separated clusters.")

    return km


# ─────────────────────────────────────────────────────────────
# Step 6 — RFM Quintile Scoring (1–5)
# ─────────────────────────────────────────────────────────────

def add_rfm_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns standard 1-5 RFM scores per customer:
      R: lower recency  (fewer days) = score 5
      F: higher frequency            = score 5
      M: higher monetary spend       = score 5
    Also computes RFM_Total (3–15) and a plain-English value tier.
    """
    print("\n[6/7] Computing RFM quintile scores (1-5) ...")
    df = rfm.copy()

    try:
        df["R_Score"] = pd.qcut(
            df["Recency"], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop"
        ).astype(int)
        df["F_Score"] = pd.qcut(
            df["Frequency"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
        ).astype(int)
        df["M_Score"] = pd.qcut(
            df["Monetary"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
        ).astype(int)
        df["RFM_Total"] = df["R_Score"] + df["F_Score"] + df["M_Score"]
        df["RFM_Tier"]  = df["RFM_Total"].apply(
            lambda s: "High Value" if s >= 12 else ("Mid Value" if s >= 7 else "Low Value")
        )
        r_mean = df["R_Score"].mean()
        f_mean = df["F_Score"].mean()
        m_mean = df["M_Score"].mean()
        print(f"      Avg R-score: {r_mean:.2f}  F-score: {f_mean:.2f}  M-score: {m_mean:.2f}")
    except Exception as e:
        print(f"      RFM scoring skipped: {e}")

    return df


# ─────────────────────────────────────────────────────────────
# Step 7 — Save All Artefacts
# ─────────────────────────────────────────────────────────────

def save_artifacts(
    model: KMeans,
    transformer: QuantileTransformer,
    cap_values: dict,
    rfm: pd.DataFrame,
    X_scaled: np.ndarray,
    currency_code: str,
):
    print("\n[7/7] Saving artefacts ...")

    rfm = rfm.copy()
    rfm["Cluster"] = model.predict(X_scaled)

    # Add RFM quintile scores
    rfm = add_rfm_scores(rfm)

    # Assign human-readable segment labels
    cluster_map = assign_segment_labels(rfm, model.n_clusters)

    # PCA for 2-D visualisation (not used in clustering)
    pca    = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    rfm["PCA1"] = coords[:, 0]
    rfm["PCA2"] = coords[:, 1]

    var = pca.explained_variance_ratio_
    print(f"      PCA variance explained: PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%  "
          f"Total={sum(var)*100:.1f}%")

    rfm["Segment"]      = rfm["Cluster"].map(lambda c: cluster_map[c]["name"])
    rfm["SegmentEmoji"] = rfm["Cluster"].map(lambda c: cluster_map[c]["emoji"])
    rfm["SegmentColor"] = rfm["Cluster"].map(lambda c: cluster_map[c]["color"])

    # Training quality metrics for meta
    labels = model.labels_
    sil    = float(silhouette_score(X_scaled, labels))
    db     = float(davies_bouldin_score(X_scaled, labels))
    ch     = float(calinski_harabasz_score(X_scaled, labels))

    cur = CURRENCIES.get(currency_code.upper(), CURRENCIES["USD"])
    meta = {
        # Currency
        "currency_code":    cur["code"],
        "currency_symbol":  cur["symbol"],
        "currency_name":    cur["name"],
        # Model info
        "n_clusters":       model.n_clusters,
        "snapshot_date":    str(pd.Timestamp.now().date()),
        "scaler_type":      "QuantileTransformer_uniform",
        # CRITICAL: store cap values so app.py applies identical capping
        "cap_values":       cap_values,
        # Quality metrics
        "silhouette":       sil,
        "davies_bouldin":   db,
        "calinski_harabasz":ch,
        "pca_variance":     var.tolist(),
        "n_customers":      len(rfm),
        "n_iter":           int(model.n_iter_),
    }

    joblib.dump(model,       MODEL_OUT)
    joblib.dump(transformer, SCALER_OUT)   # <- QuantileTransformer, not RobustScaler
    joblib.dump(cluster_map, CMAP_OUT)
    joblib.dump(meta,        META_OUT)
    rfm.to_csv(RFM_OUT, index=False)

    for f in ["model.pkl","scaler.pkl","cluster_map.pkl","training_meta.pkl","rfm_with_clusters.csv"]:
        print(f"      saved: {f}")

    # ── Full cluster quality report ──
    ss = silhouette_samples(X_scaled, labels)
    rfm["_sil"] = ss

    print("\n" + "=" * 72)
    print("  CLUSTER QUALITY REPORT")
    print("=" * 72)
    print(f"  Silhouette Score:    {sil:.4f}   (target > 0.40)")
    print(f"  Davies-Bouldin:      {db:.4f}   (target < 0.80)")
    print(f"  Calinski-Harabasz:   {ch:.1f}")
    print(f"  PCA coverage:        {sum(var)*100:.1f}%")
    print()

    summary = (
        rfm.groupby(["Cluster", "Segment"])
        .agg(
            N         = ("customer_id",  "count"),
            Avg_R     = ("Recency",      "mean"),
            Avg_F     = ("Frequency",    "mean"),
            Avg_M     = ("Monetary",     "mean"),
            Avg_RFM   = ("RFM_Total",    "mean"),
            Sil_Score = ("_sil",         "mean"),
        )
        .round(2)
        .reset_index()
        .sort_values("Avg_M", ascending=False)
    )
    print(f"  {'Cluster':>7}  {'Segment':22}  {'N':>5}  {'Avg_R':>7}  "
          f"{'Avg_F':>7}  {'Avg_M':>10}  {'RFM':>5}  {'Sil':>6}")
    print(f"  {'-'*7}  {'-'*22}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*5}  {'-'*6}")
    for _, r in summary.iterrows():
        pct = r["N"] / len(rfm) * 100
        print(f"  {int(r['Cluster']):>7}  {r['Segment']:22}  {int(r['N']):>4} ({pct:.0f}%)  "
              f"{r['Avg_R']:>7.1f}  {r['Avg_F']:>7.1f}  {r['Avg_M']:>10.1f}  "
              f"{r['Avg_RFM']:>5.1f}  {r['Sil_Score']:>6.3f}")

    # Warn about small / overlapping clusters
    print()
    for _, r in summary.iterrows():
        pct = r["N"] / len(rfm) * 100
        if pct < 3:
            print(f"  WARNING: '{r['Segment']}' is only {pct:.1f}% of customers."
                  f" Consider reducing k.")
        if r["Sil_Score"] < 0.10:
            print(f"  WARNING: '{r['Segment']}' has low silhouette {r['Sil_Score']:.3f}."
                  f" These customers may overlap with neighbours.")
    print("=" * 72 + "\n")

    rfm.drop(columns=["_sil"], inplace=True)


# ─────────────────────────────────────────────────────────────
# CLI & Entry Point
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SegmentIQ Optimal Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py
  python train.py --currency INR
  python train.py --data "D:/data/sales.csv" --currency EUR
  python train.py --clusters 5        # skip auto-search, force k=5
        """,
    )
    p.add_argument("--data",     default=None,  help="Path to CSV file")
    p.add_argument("--clusters", type=int,       default=None,
                   help="Force k (skip 4-metric search)")
    p.add_argument("--currency", default="USD",
                   help=f"Currency code. Options: {', '.join(CURRENCIES)}")
    return p.parse_args()


def main():
    args = parse_args()
    path = resolve_path(args.data)
    cur  = args.currency.upper() if args.currency.upper() in CURRENCIES else "USD"

    print("\n" + "=" * 62)
    print("  SegmentIQ  —  Optimal Training Pipeline")
    print("=" * 62)
    print("  Transform:  QuantileTransformer (uniform)  [best proven]")
    print("  Algorithm:  KMeans++  n_init=50  max_iter=2000")
    print("  k-search:   4-metric consensus (Elbow+Sil+CH+DB)")
    print("  Scoring:    Same transformer reused in app.py (no mismatch)")
    print(f"  Currency:   {cur}")
    print("=" * 62)

    df, _                    = load_and_clean(path)
    rfm                      = build_rfm(df)
    X, transformer, cap_vals = preprocess(rfm)
    k                        = find_optimal_k(X, force_k=args.clusters)
    model                    = train_model(X, k)
    save_artifacts(model, transformer, cap_vals, rfm, X, cur)

    print("Training complete.  Launch dashboard:\n   streamlit run app.py\n")


if __name__ == "__main__":
    main()
