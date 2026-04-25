"""
app.py  -  SegmentIQ Customer Segmentation Dashboard
Run:  streamlit run app.py
"""

import os, io, json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from utils import (
    clean_data, compute_rfm, validate_schema,
    assign_segment_labels, CURRENCIES, format_currency,
)

# ─────────────────────────────────────────────────────────────
# COLOR HELPER  — Plotly only accepts 6-digit hex or rgba()
# ─────────────────────────────────────────────────────────────

def hex_to_rgba(hex_color: str, alpha: float = 0.13) -> str:
    try:
        h = hex_color.strip().lstrip("#")
        if len(h) == 6:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f"rgba({r},{g},{b},{alpha})"
    except Exception:
        pass
    return f"rgba(148,163,184,{alpha})"

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────

_DIR         = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(_DIR, "model.pkl")
SCALER_PATH  = os.path.join(_DIR, "scaler.pkl")
CMAP_PATH    = os.path.join(_DIR, "cluster_map.pkl")
META_PATH    = os.path.join(_DIR, "training_meta.pkl")
RFM_CSV_PATH = os.path.join(_DIR, "sample.csv")
SAMPLE_DATA  = os.path.join(_DIR, "Data", "online_retail.csv")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SegmentIQ",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS  — pure dark theme, locked permanently
#         No light/dark switching: removes broken behaviour
#         Header clipping fix: pushes content down below toolbar
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Hard-reset everything to dark ── */
*, *::before, *::after { box-sizing: border-box; }

html, body,
.stApp,
[class*="css"],
.stApp > div,
section[data-testid="stAppViewContainer"],
section[data-testid="stAppViewContainer"] > div {
  background-color: #050c1a !important;
  color: #e2e8f0 !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ── FIX: page heading hidden behind Streamlit toolbar ──
   Push the main content down so the first element is not
   clipped by the fixed top bar                          */
.block-container {
  padding-top: 2.4rem !important;
  padding-left: 2.2rem !important;
  padding-right: 2.2rem !important;
  padding-bottom: 3rem !important;
  max-width: 1440px !important;
}
/* Kill the extra white/blank header bar entirely */
header[data-testid="stHeader"] {
  background: #050c1a !important;
  border-bottom: 1px solid #0e1e33 !important;
  height: 2.4rem !important;
}
/* Decorative top-bar ribbon Streamlit injects */
div[data-testid="stDecoration"] { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #060d1f 0%, #091525 100%) !important;
  border-right: 1px solid #162035 !important;
}
[data-testid="stSidebar"] > div { padding: 0 !important; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stRadio > label { display: none !important; }
[data-testid="stSidebar"] .stRadio label {
  display: flex !important; align-items: center !important;
  padding: 0.55rem 1rem !important; border-radius: 10px !important;
  font-size: 0.87rem !important; font-weight: 500 !important;
  cursor: pointer !important; transition: all 0.15s !important;
  border: 1px solid transparent !important; margin: 1px 0 !important;
  color: #64748b !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
  background: #162035 !important; color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
  background: #0b1729 !important; border-color: #162035 !important;
  color: #e2e8f0 !important; font-size: 0.84rem !important;
}
[data-testid="stSidebar"] .stNumberInput input {
  background: #0b1729 !important; border-color: #162035 !important;
  color: #e2e8f0 !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #94a3b8 !important; }

/* ── Streamlit widget resets to dark ── */
.stSelectbox > div > div {
  background: #0b1829 !important; border-color: #162035 !important;
  color: #e2e8f0 !important;
}
.stTextInput input, .stNumberInput input {
  background: #0b1829 !important; border-color: #162035 !important;
  color: #e2e8f0 !important;
}
.stTabs [data-baseweb="tab-list"] {
  background: #0b1829 !important; border-bottom: 1px solid #162035 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important; color: #64748b !important;
}
.stTabs [aria-selected="true"] {
  color: #38bdf8 !important; border-bottom: 2px solid #38bdf8 !important;
}
.stDataFrame, .stDataFrame * { color: #e2e8f0 !important; background: #0b1829 !important; }
div[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; }
.stButton > button {
  border-radius: 10px !important; font-weight: 700 !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  transition: opacity 0.15s, transform 0.15s !important;
}
.stButton > button:hover { opacity: 0.86 !important; transform: translateY(-1px) !important; }
.stSuccess, .stInfo, .stWarning, .stError {
  background: #0b1829 !important; color: #e2e8f0 !important;
  border-radius: 10px !important;
}

/* ── Page header ── */
.pg-header {
  display: flex; align-items: flex-start; gap: 1rem;
  margin-bottom: 1.6rem; padding-bottom: 1rem;
  border-bottom: 1px solid #162035;
}
.pg-icon {
  width: 44px; height: 44px; border-radius: 13px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.3rem; flex-shrink: 0; margin-top: 2px;
}
.pg-title { font-size: 1.5rem; font-weight: 800; color: #f8fafc; line-height: 1.15; }
.pg-sub   { font-size: 0.81rem; color: #64748b; margin-top: 0.22rem; }

/* ── KPI Cards ── */
.kpi-card {
  background: linear-gradient(145deg, #0c1a30, #071020);
  border: 1px solid #162035; border-radius: 15px;
  padding: 1.2rem 1.35rem; position: relative; overflow: hidden;
  transition: transform 0.18s, border-color 0.18s;
  border-top: 3px solid transparent;
}
.kpi-card:hover { transform: translateY(-3px); border-color: #243454; }
.kpi-ico  { position: absolute; right: 1rem; top: 1rem; font-size: 1.35rem; opacity: 0.12; }
.kpi-lbl  { color: #64748b; font-size: 0.67rem; font-weight: 700; letter-spacing: 0.11em; text-transform: uppercase; margin-bottom: 0.4rem; }
.kpi-val  { font-size: 1.85rem; font-weight: 800; line-height: 1; color: #f8fafc; }
.kpi-delta{ font-size: 0.72rem; margin-top: 0.38rem; color: #475569; }

/* ── Chart cards ── */
.chart-card {
  background: #0b1829; border: 1px solid #162035;
  border-radius: 15px; padding: 1.25rem 1.4rem; margin-bottom: 1.25rem;
}
.cc-title { font-size: 0.88rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.12rem; }
.cc-sub   { font-size: 0.72rem; color: #64748b; margin-bottom: 0.8rem; }

/* ── Section header ── */
.sec-hdr {
  font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: #334155;
  padding: 0.3rem 0; border-bottom: 1px solid #162035;
  margin: 1.4rem 0 0.95rem;
}

/* ── Segment pill ── */
.seg-pill {
  display: inline-flex; align-items: center; gap: 0.28rem;
  padding: 0.2rem 0.6rem; border-radius: 99px; font-size: 0.72rem; font-weight: 700;
}

/* ── Alert cards ── */
.alert-card {
  border-left: 4px solid; border-radius: 0 11px 11px 0;
  padding: 0.8rem 1rem; margin-bottom: 0.65rem; background: #0b1829;
}
.ac-title { font-weight: 700; font-size: 0.85rem; margin-bottom: 0.1rem; }
.ac-body  { font-size: 0.77rem; color: #94a3b8; }

/* ── Insight cards ── */
.insight-card {
  background: linear-gradient(135deg, #0b1829, #07101f);
  border: 1px solid #162035; border-left: 4px solid;
  border-radius: 0 13px 13px 0; padding: 0.95rem 1.2rem; margin-bottom: 0.7rem;
}
.ic-title  { font-weight: 700; font-size: 0.9rem; margin-bottom: 0.22rem; }
.ic-desc   { color: #94a3b8; font-size: 0.8rem; margin-bottom: 0.28rem; }
.ic-action { color: #38bdf8; font-size: 0.77rem; font-style: italic; }

/* ── Profile card ── */
.profile-card {
  background: linear-gradient(145deg, #0b1829, #060e1c);
  border: 1px solid #162035; border-radius: 18px;
  padding: 1.7rem 1.9rem; margin-bottom: 1.2rem;
}
.profile-name { font-size: 1.35rem; font-weight: 800; color: #f8fafc; margin-bottom: 0.45rem; }
.rfm-trio { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.9rem; margin: 1.3rem 0; }
.rfm-item { text-align: center; }
.rfm-val  { font-size: 1.8rem; font-weight: 800; }
.rfm-lbl  { color: #64748b; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; }
.rfm-desc { color: #334155; font-size: 0.7rem; margin-top: 0.1rem; }

/* ── Progress bars ── */
.pbar-row  { margin-bottom: 0.9rem; }
.pbar-head { display: flex; justify-content: space-between; margin-bottom: 0.25rem; }
.pbar-lbl  { font-size: 0.76rem; color: #94a3b8; }
.pbar-val  { font-size: 0.8rem; font-weight: 700; }
.pbar-track{ background: #162035; border-radius: 99px; height: 6px; overflow: hidden; }
.pbar-fill { height: 100%; border-radius: 99px; }

/* ── Info box ── */
.info-box       { background: #071020; border: 1px solid #162035; border-radius: 11px; padding: 1rem 1.2rem; margin-bottom: 1rem; }
.info-box-title { color: #38bdf8; font-weight: 700; font-size: 0.85rem; margin-bottom: 0.5rem; }
.info-box-body  { font-size: 0.78rem; color: #94a3b8; line-height: 2.1; }

/* ── Conversion card ── */
.conv-card  { background: #071020; border: 1px solid #162035; border-radius: 12px; padding: 0.9rem 1rem; margin-top: 0.5rem; }
.conv-label { color: #64748b; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.3rem; }
.conv-rate  { color: #22c55e; font-size: 1rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }

/* ── Footer ── */
.footer { margin-top: 2.5rem; padding: 1.2rem 2rem; border-top: 1px solid #162035; text-align: center; color: #334155; font-size: 0.77rem; }
.footer .name { color: #38bdf8; font-weight: 700; }

/* ── Misc ── */
.mono { font-family: 'JetBrains Mono', monospace; }
h1, h2, h3 { font-weight: 800 !important; color: #f8fafc !important; }
p { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LIVE RATE FETCH
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_rate(from_code: str, to_code: str):
    try:
        import urllib.request
        url = f"https://open.er-api.com/v6/latest/{from_code.upper()}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        rate = data.get("rates", {}).get(to_code.upper())
        return float(rate) if rate else None
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH), joblib.load(CMAP_PATH)

@st.cache_data(show_spinner=False)
def load_meta() -> dict:
    if os.path.exists(META_PATH):
        return joblib.load(META_PATH)
    return {"currency_code": "USD", "currency_symbol": "$", "currency_name": "US Dollar", "n_clusters": 5}

@st.cache_data(show_spinner=False)
def load_pretrained_rfm() -> pd.DataFrame:
    return pd.read_csv(RFM_CSV_PATH)

@st.cache_data(show_spinner="Processing your data …")
def process_bytes(raw: bytes):
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        return None, None, str(e)
    _, warn = validate_schema(df)
    df_clean, report = clean_data(df)
    rfm = compute_rfm(df_clean)
    if len(rfm) == 0:
        return None, None, "No valid customer records found in this file."
    model, scaler, cluster_map = load_model()
    meta = load_meta()
    cap_values = meta.get("cap_values", {})
    rfm_s = rfm[["Recency", "Frequency", "Monetary"]].copy()
    for col, cap in cap_values.items():
        if col in rfm_s.columns:
            rfm_s[col] = rfm_s[col].clip(upper=cap)
    X = scaler.transform(rfm_s)
    rfm["Cluster"] = model.predict(X)
    pca = PCA(n_components=2, random_state=42)
    c = pca.fit_transform(X)
    rfm["PCA1"], rfm["PCA2"] = c[:, 0], c[:, 1]
    rfm["Segment"]      = rfm["Cluster"].map(lambda x: cluster_map[x]["name"])
    rfm["SegmentEmoji"] = rfm["Cluster"].map(lambda x: cluster_map[x]["emoji"])
    rfm["SegmentColor"] = rfm["Cluster"].map(lambda x: cluster_map[x]["color"])
    return rfm, report, warn

# ─────────────────────────────────────────────────────────────
# STATE HELPERS
# ─────────────────────────────────────────────────────────────

def get_rfm():    return st.session_state.get("rfm_df")
def set_rfm(df):  st.session_state["rfm_df"] = df
def model_ok():   return os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)
def conv_rate():  return float(st.session_state.get("conv_rate", 1.0))
def conv_target():return st.session_state.get("conv_target", None)

def fmt(val: float) -> str:
    """Format monetary value with conversion rate applied."""
    return format_currency(val * conv_rate(), SYM)

def get_converted_rfm() -> pd.DataFrame | None:
    """
    Returns a COPY of the RFM dataframe with Monetary already multiplied
    by the active conversion rate.  Pass this to ALL chart functions so
    that Y-axes, hover labels and bar text automatically use converted values.
    """
    rfm = get_rfm()
    if rfm is None:
        return None
    rate = conv_rate()
    if rate == 1.0:
        return rfm.copy()
    df = rfm.copy()
    df["Monetary"] = df["Monetary"] * rate
    return df

# ─────────────────────────────────────────────────────────────
# PLOTLY DARK THEME HELPERS  — always dark regardless of OS theme
# ─────────────────────────────────────────────────────────────

_DARK_BG   = "rgba(5,12,26,1)"
_GRID      = "#1a2a3f"
_TICK      = "#64748b"
_FONT_COL  = "#e2e8f0"
_PAPER_BG  = "rgba(0,0,0,0)"

def _base_layout(**extra):
    base = dict(
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_DARK_BG,
        font_color=_FONT_COL,
        font_family="Plus Jakarta Sans",
        margin=dict(l=10, r=10, t=36, b=10),
        xaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID,
                   tickfont_color=_TICK, linecolor=_GRID),
        yaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID,
                   tickfont_color=_TICK, linecolor=_GRID),
        legend=dict(font_size=11, bgcolor="rgba(0,0,0,0)",
                    font_color=_FONT_COL, bordercolor=_GRID),
    )
    base.update(extra)
    return base

def apply(fig, **kw):
    fig.update_layout(**_base_layout(**kw))
    return fig

def polar_layout():
    return dict(
        bgcolor="rgba(5,12,26,0.95)",
        radialaxis=dict(visible=True, range=[0, 1], gridcolor=_GRID,
                        tickfont_color=_TICK, tickfont_size=9),
        angularaxis=dict(gridcolor=_GRID, tickfont_color="#94a3b8",
                         tickfont_size=10),
    )

def dark_pie(fig):
    fig.update_layout(
        paper_bgcolor=_PAPER_BG, font_color=_FONT_COL,
        font_family="Plus Jakarta Sans",
        legend=dict(font_size=10, bgcolor="rgba(0,0,0,0)",
                    font_color=_FONT_COL),
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig

# ─────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────

def page_header(icon, bg, title, sub):
    st.markdown(f"""
    <div class="pg-header">
      <div class="pg-icon" style="background:{bg}20;border:1px solid {bg}40;">{icon}</div>
      <div>
        <div class="pg-title">{title}</div>
        <div class="pg-sub">{sub}</div>
      </div>
    </div>""", unsafe_allow_html=True)

def kpi(label, value, delta, icon, accent):
    st.markdown(f"""
    <div class="kpi-card" style="border-top-color:{accent};">
      <div class="kpi-ico">{icon}</div>
      <div class="kpi-lbl">{label}</div>
      <div class="kpi-val" style="color:{accent};">{value}</div>
      <div class="kpi-delta">{delta}</div>
    </div>""", unsafe_allow_html=True)

def sec(text):
    st.markdown(f'<div class="sec-hdr">{text}</div>', unsafe_allow_html=True)

def card(title, sub=""):
    st.markdown(
        f'<div class="chart-card"><div class="cc-title">{title}</div>'
        + (f'<div class="cc-sub">{sub}</div>' if sub else ""),
        unsafe_allow_html=True,
    )

def end_card():
    st.markdown("</div>", unsafe_allow_html=True)

def footer():
    st.markdown(
        '<div class="footer">Developed by &nbsp;'
        '<span class="name">Surya Prasad Yadav</span>'
        '&nbsp;&middot;&nbsp;<b>Data Science Engineer</b></div>',
        unsafe_allow_html=True,
    )

def _safe_radar(rfm_c, color_map):
    """Build radar using CONVERTED rfm so axes reflect active currency."""
    try:
        rfm_n = rfm_c[["Segment","Recency","Frequency","Monetary"]].copy()
        for col in ["Recency","Frequency","Monetary"]:
            mn, mx = rfm_n[col].min(), rfm_n[col].max()
            denom = mx - mn if mx != mn else 1
            rfm_n[col] = (rfm_n[col] - mn) / denom
        rfm_n["Recency"] = 1 - rfm_n["Recency"]
        radar = rfm_n.groupby("Segment")[["Recency","Frequency","Monetary"]].mean().reset_index()
        cats  = ["Recency Score","Frequency Score","Revenue Score"]
        fig   = go.Figure()
        for _, row in radar.iterrows():
            c = color_map.get(row["Segment"], "#94a3b8")
            v = [float(row["Recency"]), float(row["Frequency"]), float(row["Monetary"])]
            fig.add_trace(go.Scatterpolar(
                r=v+[v[0]], theta=cats+[cats[0]],
                fill="toself", name=row["Segment"],
                line=dict(color=c, width=2),
                fillcolor=hex_to_rgba(c, 0.13),
            ))
        fig.update_layout(
            polar=polar_layout(),
            paper_bgcolor=_PAPER_BG, font_color=_FONT_COL,
            font_family="Plus Jakarta Sans",
            legend=dict(font_size=10, bgcolor="rgba(0,0,0,0)", font_color=_FONT_COL),
            margin=dict(l=10, r=10, t=10, b=10), height=360,
        )
        return fig
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:1.4rem 1.2rem 0.9rem;border-bottom:1px solid #162035;">
      <div style="display:flex;align-items:center;gap:0.65rem;margin-bottom:0.22rem;">
        <div style="width:31px;height:31px;
                    background:linear-gradient(135deg,#1d4ed8,#0ea5e9);
                    border-radius:9px;display:flex;align-items:center;
                    justify-content:center;font-size:0.95rem;">⬡</div>
        <span style="font-size:1.18rem;font-weight:800;color:#f8fafc;">SegmentIQ</span>
      </div>
      <div style="color:#334155;font-size:0.7rem;padding-left:2.5rem;">
        Customer Intelligence Platform
      </div>
    </div>""", unsafe_allow_html=True)

    # Navigation
    st.markdown("<div style='padding:0.6rem 1rem 0.1rem;'>"
                "<div class='sec-hdr'>Pages</div></div>",
                unsafe_allow_html=True)
    page = st.radio("", [
        "🏠  Overview",
        "📂  Data Input",
        "🔢  Segmentation",
        "📊  Visualisation",
        "💡  Business Insights",
        "🔍  Customer Lookup",
    ], label_visibility="collapsed")

    # ── Currency Symbol (display only, no conversion) ──
    st.markdown("<div style='padding:0.4rem 1rem 0.1rem;'>"
                "<div class='sec-hdr'>Currency Symbol</div></div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='padding:0 0.7rem 0.3rem;color:#475569;font-size:0.68rem;"
        "line-height:1.5;'>Changes the symbol shown next to values.<br>"
        "Does not convert amounts — use Currency Conversion below for that.</div>",
        unsafe_allow_html=True,
    )
    meta     = load_meta()
    cur_opts = {f"{v['code']}  {v['symbol']}  —  {v['name']}": k for k, v in CURRENCIES.items()}
    def_lbl  = next((l for l, c in cur_opts.items()
                     if c == meta.get("currency_code", "USD")), list(cur_opts.keys())[0])
    sel_lbl  = st.selectbox("cur", list(cur_opts.keys()),
                            index=list(cur_opts.keys()).index(def_lbl),
                            label_visibility="collapsed")
    CUR  = CURRENCIES[cur_opts[sel_lbl]]
    SYM  = CUR["symbol"]
    CODE = CUR["code"]

    # ── Currency Conversion (multiplies values, fully separate) ──
    st.markdown("<div style='padding:0.4rem 1rem 0.1rem;'>"
                "<div class='sec-hdr' style='color:#22c55e55;'>"
                "Currency Conversion <span style='color:#22c55e;font-size:0.6rem;'></span>"
                "</div></div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='padding:0 0.7rem 0.3rem;color:#475569;font-size:0.68rem;"
        "line-height:1.5;'>Multiply ALL monetary values by a rate.<br>"
        "Set manually or fetch live from the internet.</div>",
        unsafe_allow_html=True,
    )

    # Target currency selector
    conv_opts   = {f"{v['code']}  {v['symbol']}  —  {v['name']}": k for k, v in CURRENCIES.items()}
    conv_tgt_lb = st.selectbox("Convert to", list(conv_opts.keys()),
                               index=0, label_visibility="visible",
                               key="conv_target_select")
    conv_tgt    = conv_opts[conv_tgt_lb]

    # Rate input + live fetch button
    col_inp, col_btn = st.columns([3, 2])
    with col_inp:
        manual_rate = st.number_input(
            "Rate", min_value=0.0001, max_value=1_000_000.0,
            value=float(st.session_state.get("conv_rate", 1.0)),
            step=0.01, format="%.4f",
            label_visibility="visible", key="manual_rate_input",
        )
    with col_btn:
        st.markdown("<div style='margin-top:1.65rem;'></div>", unsafe_allow_html=True)
        if st.button("🌐 Live", use_container_width=True, key="fetch_rate_btn"):
            with st.spinner("Fetching …"):
                live = fetch_live_rate(CODE, conv_tgt)
            if live:
                st.session_state["conv_rate"]   = live
                st.session_state["conv_target"] = conv_tgt
                st.rerun()
            else:
                st.warning("Could not fetch. Enter manually.")

    # Apply / Reset
    ca, cb = st.columns(2)
    with ca:
        if st.button("✅ Apply", use_container_width=True, key="apply_btn"):
            st.session_state["conv_rate"]   = manual_rate
            st.session_state["conv_target"] = conv_tgt
            st.rerun()
    with cb:
        if st.button("🔄 Reset", use_container_width=True, key="reset_btn"):
            st.session_state["conv_rate"]   = 1.0
            st.session_state["conv_target"] = None
            st.rerun()

    # Active conversion indicator
    active_rate   = conv_rate()
    active_target = conv_target()
    if active_rate != 1.0 and active_target:
        st.markdown(f"""
        <div class="conv-card">
          <div class="conv-label">Active Conversion</div>
          <div class="conv-rate">1 {CODE} = {active_rate:.4f} {active_target}</div>
          <div style="color:#334155;font-size:0.68rem;margin-top:0.2rem;">
            All monetary values × {active_rate:.4f}
          </div>
        </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# PAGE 0 — OVERVIEW
# ═════════════════════════════════════════════════════════════

def page_overview():
    page_header("🏠","#38bdf8","Dashboard Overview",
                "Executive summary — revenue health, segment pulse & quick-win opportunities")
    rfm_c = get_converted_rfm()

    if rfm_c is None:
        st.info("👋 Welcome to **SegmentIQ**. Upload your data in **Data Input** to get started.")
        cols = st.columns(4)
        feats = [
            ("🔢","#22c55e","RFM Segmentation",  "Recency · Frequency · Monetary"),
            ("⚡","#38bdf8","Zero Retraining",    "Pre-trained model scores instantly"),
            ("📊","#f59e0b","Deep Visualisations","PCA, 3-D, heatmaps, radar"),
            ("💡","#8b5cf6","Actionable Insights", "Plain-English strategy per segment"),
        ]
        for col,(ico,c,t,d) in zip(cols,feats):
            with col:
                st.markdown(f"""
                <div class="kpi-card" style="border-top-color:{c};text-align:center;padding:1.5rem 1rem;">
                  <div style="font-size:1.9rem;margin-bottom:0.55rem;">{ico}</div>
                  <div style="font-weight:700;color:#f8fafc;font-size:0.9rem;margin-bottom:0.22rem;">{t}</div>
                  <div style="color:#64748b;font-size:0.75rem;">{d}</div>
                </div>""", unsafe_allow_html=True)
        footer(); return

    _, _, cluster_map = load_model()
    # All aggregation on CONVERTED rfm — graphs automatically correct
    seg_stats = (rfm_c.groupby(["Segment","SegmentColor"])
        .agg(Customers=("customer_id","count"), Revenue=("Monetary","sum"),
             Avg_R=("Recency","mean"), Avg_F=("Frequency","mean"), Avg_M=("Monetary","mean"))
        .reset_index().sort_values("Revenue", ascending=False))
    total_rev = rfm_c["Monetary"].sum()
    champ     = seg_stats[seg_stats["Segment"]=="Champions"]
    champ_rev = float(champ["Revenue"].values[0]) if len(champ) else 0.0
    champ_n   = int(champ["Customers"].values[0]) if len(champ) else 0

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Total Customers",       f"{len(rfm_c):,}",
                 f"{rfm_c['Segment'].nunique()} segments","👥","#38bdf8")
    with c2: kpi(f"Total Revenue ({CODE})",
                 format_currency(total_rev, SYM),"Lifetime value","💰","#22c55e")
    with c3: kpi(f"Avg Revenue ({CODE})",
                 format_currency(rfm_c["Monetary"].mean(), SYM),
                 f"Median {format_currency(rfm_c['Monetary'].median(), SYM)}","🧾","#f59e0b")
    with c4: kpi("Champions Share",
                 f"{champ_rev/total_rev*100:.1f}%",
                 f"{champ_n} customers drive it","🏆","#8b5cf6")

    st.markdown("<br>", unsafe_allow_html=True)
    col1,col2 = st.columns([3,2])

    with col1:
        card(f"Revenue by Segment ({CODE})","Total lifetime value per cluster")
        cmap = dict(zip(seg_stats["Segment"], seg_stats["SegmentColor"]))
        # Y-axis and bar text both use converted Revenue column
        fig = px.bar(
            seg_stats, x="Segment", y="Revenue", color="Segment",
            color_discrete_map=cmap,
            text=seg_stats["Revenue"].apply(
                lambda v: format_currency(v, SYM)),
        )
        fig.update_traces(marker_line_width=0, textposition="outside")
        apply(fig, showlegend=False)
        fig.update_yaxes(tickprefix=SYM, tickformat=",.0f",
                         gridcolor=_GRID, tickfont_color=_TICK)
        st.plotly_chart(fig, use_container_width=True)
        end_card()

    with col2:
        card("Segment Composition","Customer headcount share")
        fig2 = px.pie(
            seg_stats, names="Segment", values="Customers", color="Segment",
            color_discrete_map=dict(zip(seg_stats["Segment"],seg_stats["SegmentColor"])),
            hole=0.62,
        )
        dark_pie(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        end_card()

    sec("🚨 Quick-Win Opportunities")
    ca,cb = st.columns(2)
    alerts = [
        ("At-Risk Customers","⚠️","#f97316",
         "Previously active buyers who've gone quiet.",
         "Launch a personalised 20% win-back campaign."),
        ("Lost / Inactive","❌","#ef4444",
         "Long-dormant — reactivation cost is high.",
         "A/B test a 'We miss you' email vs a sunset offer."),
        ("New Customers","🌱","#06b6d4",
         "First 30 days are critical for long-term retention.",
         "Trigger a 3-email onboarding sequence immediately."),
        ("Potential Loyalists","🔮","#8b5cf6",
         "Repeat-buying signals — prime for loyalty upgrade.",
         "Invite to VIP tier before a competitor does."),
    ]
    for i,(seg,emoji,color,desc,action) in enumerate(alerts):
        (ca if i%2==0 else cb).markdown(f"""
        <div class="alert-card" style="border-left-color:{color};">
          <div class="ac-title" style="color:{color};">{emoji} {seg}</div>
          <div class="ac-body">{desc}<br>
          <span style="color:#38bdf8;font-style:italic;font-size:0.76rem;">
            → {action}</span></div>
        </div>""", unsafe_allow_html=True)

    sec("📐 RFM Benchmark by Segment")
    strip_cols = st.columns(len(seg_stats))
    for col,(_,row) in zip(strip_cols, seg_stats.iterrows()):
        with col:
            st.markdown(f"""
            <div style="background:#0b1829;border:1px solid #162035;
                        border-top:3px solid {row['SegmentColor']};border-radius:12px;
                        padding:0.85rem 0.65rem;text-align:center;">
              <div style="font-size:1rem;margin-bottom:0.25rem;">
                {row.get('SegmentEmoji','⬡')}</div>
              <div style="font-weight:700;font-size:0.77rem;color:{row['SegmentColor']};
                          margin-bottom:0.4rem;">{row['Segment']}</div>
              <div style="font-size:0.68rem;color:#64748b;">
                R <b style="color:#94a3b8">{row['Avg_R']:.0f}d</b></div>
              <div style="font-size:0.68rem;color:#64748b;">
                F <b style="color:#94a3b8">{row['Avg_F']:.1f}</b></div>
              <div style="font-size:0.68rem;color:#64748b;">
                M <b style="color:#94a3b8">{format_currency(row['Avg_M'], SYM)}</b></div>
            </div>""", unsafe_allow_html=True)
    footer()


# ═════════════════════════════════════════════════════════════
# PAGE 1 — DATA INPUT
# ═════════════════════════════════════════════════════════════

def page_data_input():
    page_header("📂","#38bdf8","Data Input",
                "Upload any CSV — auto-detects columns, imputes missing values, reads ALL rows")
    if not model_ok():
        st.error("🚨 Run `python train.py` first."); footer(); return

    tab_upload, tab_sample = st.tabs(["⬆️  Upload CSV","🗂  Sample Dataset"])

    with tab_upload:
        st.markdown("""
        <div class="info-box">
          <div class="info-box-title">🔍 Smart Column Detection — Full Data Reading</div>
          <div class="info-box-body">
            Every row and every unique customer is processed — no row limit.<br>
            <b style="color:#e2e8f0">Customer</b> — customer_id, cust_id, client_id, user_id …<br>
            <b style="color:#e2e8f0">Invoice</b> — invoice_id, order_id, transaction_id, receipt_id …<br>
            <b style="color:#e2e8f0">Date</b> — invoice_date, order_date, date, timestamp …<br>
            <b style="color:#e2e8f0">Quantity</b> — quantity, qty, units, item_count …<br>
            <b style="color:#e2e8f0">Price</b> — unit_price, price, rate, cost, selling_price …<br>
            <span style="color:#475569;">Missing columns/values are imputed automatically.</span>
          </div>
        </div>""", unsafe_allow_html=True)

        uploaded = st.file_uploader("Drop your CSV", type=["csv"],
                                    label_visibility="collapsed")
        if uploaded:
            with st.spinner("Reading all rows, detecting columns, computing RFM …"):
                rfm, report, warn = process_bytes(uploaded.read())
            if rfm is None:
                st.error(f"❌ {warn}")
            else:
                set_rfm(rfm)
                st.success(f"✅ **{len(rfm):,}** unique customers segmented")
                if warn:     st.info(f"ℹ️ {warn}")
                if report:
                    parts = []
                    inf = report.get("inferred",{})
                    imp = {k:v for k,v in report.get("imputed",{}).items() if v>0}
                    drp = {k:v for k,v in report.get("dropped",{}).items() if v>0}
                    if inf:  parts.append(f"Columns mapped: {inf}")
                    if imp:  parts.append(f"Imputed: {imp}")
                    if drp:  parts.append(f"Dropped: {drp}")
                    if parts: st.info("🔧 " + " · ".join(parts))

    with tab_sample:
        st.info("Bundled demo — 1,000 customers · all transactions included")
        if st.button("🚀 Load Sample Dataset", type="primary"):
            set_rfm(load_pretrained_rfm())
            st.success("✅ Sample loaded")
        if os.path.exists(SAMPLE_DATA):
            st.dataframe(pd.read_csv(SAMPLE_DATA).head(8), use_container_width=True)

    footer()


# ═════════════════════════════════════════════════════════════
# PAGE 2 — SEGMENTATION
# ═════════════════════════════════════════════════════════════

def page_segmentation():
    page_header("🔢","#22c55e","Customer Segmentation",
                "KMeans clusters on scaled RFM — pre-trained model, no retraining")
    rfm_c = get_converted_rfm()
    if rfm_c is None:
        st.warning("⚠️ Load data in **Data Input** first."); footer(); return

    model,_,_ = load_model()
    dist = rfm_c.groupby(["Segment","SegmentColor"]).size().reset_index(name="Count")

    c1,c2,c3,c4 = st.columns(4)
    with c1: kpi("Customers",  f"{len(rfm_c):,}","unique IDs","👥","#38bdf8")
    with c2: kpi("Segments",   str(model.n_clusters),"KMeans clusters","⬡","#22c55e")
    with c3: kpi("Avg Recency",f"{rfm_c['Recency'].mean():.0f}d",
                 f"Median {rfm_c['Recency'].median():.0f}d","📅","#f59e0b")
    with c4: kpi(f"Avg Revenue ({CODE})",
                 format_currency(rfm_c["Monetary"].mean(), SYM),
                 f"Median {format_currency(rfm_c['Monetary'].median(), SYM)}",
                 "💰","#8b5cf6")

    st.markdown("<br>", unsafe_allow_html=True)
    col1,col2 = st.columns([3,2])
    with col1:
        card("Customer Count by Segment","Headcount per cluster")
        fig = px.bar(dist, x="Segment", y="Count", color="Segment",
                     color_discrete_map=dict(zip(dist["Segment"],dist["SegmentColor"])),
                     text="Count")
        fig.update_traces(marker_line_width=0, textposition="outside")
        apply(fig, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        end_card()
    with col2:
        card("Segment Share")
        fig2 = px.pie(dist, names="Segment", values="Count", color="Segment",
                      color_discrete_map=dict(zip(dist["Segment"],dist["SegmentColor"])),
                      hole=0.6)
        dark_pie(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        end_card()

    sec("🌡️ Cluster Centroid Heatmap")
    card("Normalised RFM Scores per Segment",
         "0 = lowest · 1 = highest · Recency inverted (fewer days = better)")
    # Heatmap uses normalised scores — conversion doesn't change shape, only scale
    centroids = rfm_c.groupby("Segment")[["Recency","Frequency","Monetary"]].mean()
    norm = centroids.copy()
    for col in ["Recency","Frequency","Monetary"]:
        mn,mx = norm[col].min(), norm[col].max()
        norm[col] = (norm[col]-mn) / (mx-mn+1e-9)
    norm["Recency"] = 1 - norm["Recency"]
    fig_hm = px.imshow(norm.round(2), text_auto=".2f", aspect="auto",
                       color_continuous_scale="Blues")
    fig_hm.update_layout(paper_bgcolor=_PAPER_BG, plot_bgcolor=_PAPER_BG,
                         font_color=_FONT_COL, font_family="Plus Jakarta Sans",
                         margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig_hm, use_container_width=True)
    end_card()

    sec("📋 Full RFM Feature Table")
    show = rfm_c[["customer_id","Recency","Frequency","Monetary","Segment"]].copy()
    show = show.sort_values("Monetary", ascending=False).reset_index(drop=True)
    show["Monetary"] = show["Monetary"].apply(lambda v: format_currency(v, SYM))
    show.rename(columns={"customer_id":"Customer ID","Recency":"Recency (days)",
                          "Monetary":f"Revenue ({CODE})"}, inplace=True)
    st.dataframe(show, use_container_width=True, height=380)
    footer()


# ═════════════════════════════════════════════════════════════
# PAGE 3 — VISUALISATION
# ═════════════════════════════════════════════════════════════

def page_visualisation():
    page_header("📊","#f59e0b","Visualisation",
                "PCA cluster map · violin plots · 3-D scatter · correlation matrix")
    rfm_c = get_converted_rfm()
    if rfm_c is None:
        st.warning("⚠️ Load data in **Data Input** first."); footer(); return

    color_map = rfm_c.set_index("Segment")["SegmentColor"].to_dict()

    # PCA — axes are PC components, not monetary, so no currency on axes
    card("2-D PCA Cluster Map",
         "RFM space projected to 2 principal components · hover for details")
    fig_pca = px.scatter(
        rfm_c, x="PCA1", y="PCA2", color="Segment",
        color_discrete_map=color_map, opacity=0.82,
        hover_data={"customer_id":True,"Recency":True,"Frequency":True,
                    "Monetary":True,"PCA1":False,"PCA2":False},
        labels={"PCA1":"PC 1","PCA2":"PC 2","Monetary":f"Revenue ({CODE})"},
    )
    fig_pca.update_traces(marker=dict(size=7,
                          line=dict(width=0.4, color="rgba(255,255,255,0.1)")))
    apply(fig_pca, height=460)
    # Monetary in hover uses converted value already (rfm_c)
    fig_pca.update_traces(
        customdata=rfm_c[["customer_id","Recency","Frequency","Monetary"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Recency: %{customdata[1]:.0f}d<br>"
            "Frequency: %{customdata[2]:.0f}<br>"
            f"Revenue: {SYM}%{{customdata[3]:,.0f}}<extra></extra>"
        ),
    )
    st.plotly_chart(fig_pca, use_container_width=True)
    end_card()

    # Violin plots — Monetary uses converted rfm_c
    sec("🎻 RFM Distribution by Segment")
    c1,c2,c3 = st.columns(3)
    for col,feat,lbl in zip([c1,c2,c3],
                             ["Recency","Frequency","Monetary"],
                             ["days","transactions", f"{CODE}"]):
        with col:
            card(feat, lbl)
            fig = px.violin(rfm_c, x="Segment", y=feat, color="Segment",
                            color_discrete_map=color_map, box=True, points="outliers")
            fig.update_layout(
                paper_bgcolor=_PAPER_BG, plot_bgcolor=_DARK_BG,
                font_color=_FONT_COL, font_family="Plus Jakarta Sans",
                showlegend=False, margin=dict(l=0,r=0,t=10,b=0),
                xaxis=dict(tickfont_size=8, gridcolor=_GRID),
                yaxis=dict(gridcolor=_GRID,
                           tickprefix=SYM if feat=="Monetary" else ""),
            )
            st.plotly_chart(fig, use_container_width=True)
            end_card()

    # 3-D scatter — Z axis is Monetary from converted rfm_c
    card(f"3-D RFM Customer Space",
         f"Rotate to explore cluster separation · Z = Revenue ({CODE})")
    fig_3d = px.scatter_3d(
        rfm_c, x="Recency", y="Frequency", z="Monetary",
        color="Segment", color_discrete_map=color_map,
        opacity=0.78, hover_data=["customer_id"],
        labels={"Monetary": f"Revenue ({CODE})"},
    )
    fig_3d.update_traces(marker=dict(size=4))
    fig_3d.update_layout(
        paper_bgcolor=_PAPER_BG, font_color=_FONT_COL,
        font_family="Plus Jakarta Sans", height=520,
        scene=dict(
            bgcolor=_DARK_BG,
            xaxis=dict(gridcolor=_GRID, color=_TICK, title="Recency (days)"),
            yaxis=dict(gridcolor=_GRID, color=_TICK, title="Frequency"),
            zaxis=dict(gridcolor=_GRID, color=_TICK,
                       title=f"Revenue ({CODE})", tickprefix=SYM),
        ),
        margin=dict(l=0,r=0,t=10,b=0),
        legend=dict(font_size=11, bgcolor="rgba(0,0,0,0)", font_color=_FONT_COL),
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    end_card()

    # Correlation — structure unchanged by currency, show labels correctly
    sec("🔗 RFM Correlation Matrix")
    card("Pairwise Pearson Correlation",
         "Reveals natural RFM interdependencies (conversion doesn't affect correlation)")
    corr = rfm_c[["Recency","Frequency","Monetary"]].corr()
    corr.index   = ["Recency","Frequency",f"Revenue ({CODE})"]
    corr.columns = ["Recency","Frequency",f"Revenue ({CODE})"]
    fig_corr = px.imshow(corr.round(2), text_auto=".2f", aspect="auto",
                         color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig_corr.update_layout(paper_bgcolor=_PAPER_BG, plot_bgcolor=_PAPER_BG,
                           font_color=_FONT_COL, font_family="Plus Jakarta Sans",
                           margin=dict(l=10,r=10,t=10,b=10), height=300)
    st.plotly_chart(fig_corr, use_container_width=True)
    end_card()
    footer()


# ═════════════════════════════════════════════════════════════
# PAGE 4 — BUSINESS INSIGHTS
# ═════════════════════════════════════════════════════════════

def page_business_insights():
    page_header("💡","#8b5cf6","Business Insights",
                "Revenue · CLV · RFM radar · Pareto funnel · segment action cards")
    rfm_c = get_converted_rfm()
    if rfm_c is None:
        st.warning("⚠️ Load data in **Data Input** first."); footer(); return

    _,_,cluster_map = load_model()
    color_map = rfm_c.set_index("Segment")["SegmentColor"].to_dict()
    total_rev = rfm_c["Monetary"].sum()

    seg_stats = (rfm_c.groupby(["Segment","SegmentColor"])
        .agg(Customers=("customer_id","count"),
             Total_Rev=("Monetary","sum"),   Avg_Rev=("Monetary","mean"),
             Avg_Rec=("Recency","mean"),      Avg_Frq=("Frequency","mean"),
             Med_Rev=("Monetary","median"))
        .reset_index().sort_values("Total_Rev", ascending=False))
    seg_stats["Rev_Share"] = (seg_stats["Total_Rev"]/total_rev*100).round(1)
    seg_stats["CLV_Score"] = (
        (seg_stats["Avg_Rev"] * seg_stats["Avg_Frq"]) / (seg_stats["Avg_Rec"]+1)
    ).round(1)

    col1,col2 = st.columns(2)
    with col1:
        card(f"Total Revenue by Segment ({CODE})","Revenue share % labelled on bars")
        fig = px.bar(
            seg_stats, x="Segment", y="Total_Rev", color="Segment",
            color_discrete_map=dict(zip(seg_stats["Segment"],seg_stats["SegmentColor"])),
            text="Rev_Share",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside",
                          marker_line_width=0)
        apply(fig, showlegend=False)
        # Y-axis ticks use converted values (Total_Rev is already converted)
        fig.update_yaxes(tickprefix=SYM, tickformat=",.0f",
                         gridcolor=_GRID, tickfont_color=_TICK)
        # Hover shows converted revenue
        fig.update_traces(
            customdata=seg_stats[["Total_Rev","Rev_Share"]].values,
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"Revenue: {SYM}%{{customdata[0]:,.0f}}<br>"
                "Share: %{customdata[1]:.1f}%<extra></extra>"
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
        end_card()

    with col2:
        card("Customer Lifetime Value Score","(Avg Revenue × Frequency) ÷ Recency")
        fig2 = px.bar(
            seg_stats, x="Segment", y="CLV_Score", color="Segment",
            color_discrete_map=dict(zip(seg_stats["Segment"],seg_stats["SegmentColor"])),
            text="CLV_Score",
        )
        fig2.update_traces(texttemplate="%{text:.0f}", textposition="outside",
                           marker_line_width=0)
        apply(fig2, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        end_card()

    col3,col4 = st.columns(2)
    with col3:
        card("RFM Radar — Normalised Segment Profiles","Higher = better · Recency inverted")
        fig_r = _safe_radar(rfm_c, color_map)
        if fig_r:
            st.plotly_chart(fig_r, use_container_width=True)
        else:
            st.info("Requires at least 2 segments.")
        end_card()

    with col4:
        card("Revenue Treemap","Area proportional to segment revenue")
        fig_tm = px.treemap(
            seg_stats, path=["Segment"], values="Total_Rev", color="Segment",
            color_discrete_map=dict(zip(seg_stats["Segment"],seg_stats["SegmentColor"])),
        )
        fig_tm.update_layout(paper_bgcolor=_PAPER_BG, font_color=_FONT_COL,
                             font_family="Plus Jakarta Sans",
                             margin=dict(l=0,r=0,t=0,b=0), height=360)
        fig_tm.update_traces(textfont_size=13, textfont_color="#f8fafc")
        st.plotly_chart(fig_tm, use_container_width=True)
        end_card()

    sec("💸 Revenue Concentration — Pareto Funnel")
    card("Cumulative Revenue (Best → Worst Segment)",
         "Shows how few segments drive the majority of value")
    s_sorted = seg_stats.sort_values("Total_Rev", ascending=False)
    cum_pct  = (s_sorted["Total_Rev"].cumsum() / total_rev * 100).round(1)
    fig_par  = go.Figure()
    fig_par.add_trace(go.Bar(
        x=s_sorted["Segment"].tolist(), y=s_sorted["Rev_Share"].tolist(),
        name="Segment Rev %",
        marker_color=[r["SegmentColor"] for _,r in s_sorted.iterrows()],
        opacity=0.65,
    ))
    fig_par.add_trace(go.Scatter(
        x=s_sorted["Segment"].tolist(), y=cum_pct.tolist(),
        mode="lines+markers+text", name="Cumulative %",
        line=dict(color="#22c55e", width=2.5), marker=dict(size=9),
        text=[f"{v:.0f}%" for v in cum_pct], textposition="top center",
        textfont=dict(color="#22c55e", size=11),
    ))
    apply(fig_par, height=360, showlegend=True,
          yaxis=dict(title="Revenue Share %", gridcolor=_GRID, tickfont_color=_TICK))
    st.plotly_chart(fig_par, use_container_width=True)
    end_card()

    sec("📊 Segment Summary")
    disp = seg_stats.copy()
    disp["Total_Rev"] = disp["Total_Rev"].apply(lambda v: format_currency(v, SYM))
    disp["Avg_Rev"]   = disp["Avg_Rev"].apply(lambda v: format_currency(v, SYM))
    disp["Med_Rev"]   = disp["Med_Rev"].apply(lambda v: format_currency(v, SYM))
    disp["Avg_Rec"]   = disp["Avg_Rec"].apply(lambda v: f"{v:.0f}d")
    disp["Rev_Share"] = disp["Rev_Share"].apply(lambda v: f"{v:.1f}%")
    disp["CLV_Score"] = disp["CLV_Score"].apply(lambda v: f"{v:.1f}")
    disp.rename(columns={
        "Total_Rev":f"Total ({CODE})","Avg_Rev":"Avg Rev","Med_Rev":"Median",
        "Avg_Rec":"Avg Recency","Avg_Frq":"Avg Freq",
        "Rev_Share":"Rev %","CLV_Score":"CLV",
    }, inplace=True)
    st.dataframe(disp[["Segment","Customers",f"Total ({CODE})","Avg Rev",
                        "Median","Avg Recency","Avg Freq","Rev %","CLV"]],
                 use_container_width=True)

    sec("🎯 Segment Interpretation & Recommended Actions")
    visible     = rfm_c["Segment"].unique()
    seg_rev_map = dict(zip(seg_stats["Segment"], seg_stats["Total_Rev"].astype(float)))
    shown = sorted(
        [i for i in cluster_map.values() if i["name"] in visible],
        key=lambda x: seg_rev_map.get(x["name"],0), reverse=True,
    )
    ca,cb = st.columns(2)
    for i,info in enumerate(shown):
        (ca if i%2==0 else cb).markdown(f"""
        <div class="insight-card" style="border-left-color:{info['color']};">
          <div class="ic-title" style="color:{info['color']};">{info['emoji']} {info['name']}</div>
          <div class="ic-desc">{info['description']}</div>
          <div class="ic-action">💼 {info['action']}</div>
        </div>""", unsafe_allow_html=True)
    footer()


# ═════════════════════════════════════════════════════════════
# PAGE 5 — CUSTOMER LOOKUP
# ═════════════════════════════════════════════════════════════

def page_customer_lookup():
    page_header("🔍","#06b6d4","Customer Lookup",
                "Search any customer — profile · percentile rankings · peer comparison")
    rfm_c = get_converted_rfm()
    if rfm_c is None:
        st.warning("⚠️ Load data in **Data Input** first."); footer(); return

    _,_,cluster_map = load_model()
    color_map = rfm_c.set_index("Segment")["SegmentColor"].to_dict()
    ids = sorted(rfm_c["customer_id"].astype(str).unique().tolist())

    cs,cb = st.columns([5,1])
    with cs:
        search_id = st.selectbox("Customer ID", [""]+ids, index=0,
                                  label_visibility="collapsed",
                                  placeholder="Select or type a Customer ID…")
    with cb:
        if st.button("🎲 Random", use_container_width=True):
            st.session_state["__cid"] = np.random.choice(ids)

    if not search_id and "__cid" in st.session_state:
        search_id = st.session_state["__cid"]

    if not search_id:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;">
          <div style="font-size:3rem;margin-bottom:0.8rem;">🔍</div>
          <div style="font-size:0.98rem;font-weight:600;color:#64748b;">
            Select a customer to view their profile</div>
          <div style="font-size:0.8rem;margin-top:0.3rem;color:#334155;">
            or click 🎲 Random</div>
        </div>""", unsafe_allow_html=True)
        footer(); return

    row = rfm_c[rfm_c["customer_id"].astype(str)==str(search_id)]
    if row.empty:
        st.error(f"Customer **{search_id}** not found."); footer(); return

    row      = row.iloc[0]
    seg_info = cluster_map.get(int(row["Cluster"]), {})
    color    = seg_info.get("color","#64748b")

    st.markdown(f"""
    <div class="profile-card">
      <div style="display:flex;justify-content:space-between;
                  align-items:flex-start;flex-wrap:wrap;gap:0.9rem;">
        <div>
          <div class="profile-name">Customer {search_id}</div>
          <span class="seg-pill"
            style="background:{color}1a;color:{color};border:1px solid {color}44;">
            {seg_info.get('emoji','📦')} {seg_info.get('name','Unknown')}
          </span>
        </div>
        <div style="text-align:right;">
          <div style="color:#334155;font-size:0.65rem;text-transform:uppercase;
                      letter-spacing:0.09em;">Cluster</div>
          <div class="mono" style="font-size:1.5rem;font-weight:800;color:{color};">
            #{int(row['Cluster'])}</div>
        </div>
      </div>
      <div class="rfm-trio">
        <div class="rfm-item">
          <div class="rfm-val" style="color:{color};">{int(row['Recency'])}d</div>
          <div class="rfm-lbl">Recency</div><div class="rfm-desc">days since last purchase</div>
        </div>
        <div class="rfm-item">
          <div class="rfm-val" style="color:{color};">{int(row['Frequency'])}</div>
          <div class="rfm-lbl">Frequency</div><div class="rfm-desc">unique transactions</div>
        </div>
        <div class="rfm-item">
          <div class="rfm-val" style="color:{color};">{format_currency(row['Monetary'],SYM)}</div>
          <div class="rfm-lbl">Revenue ({CODE})</div><div class="rfm-desc">total lifetime spend</div>
        </div>
      </div>
      <hr style="border:none;border-top:1px solid #162035;margin:0.9rem 0;">
      <p style="color:#94a3b8;font-size:0.84rem;margin-bottom:0.3rem;">
        {seg_info.get('description','')}</p>
      <p style="color:#38bdf8;font-size:0.81rem;font-style:italic;">
        💼 {seg_info.get('action','')}</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # All percentile / normalisation calculations on CONVERTED rfm_c
    all_min = rfm_c[["Recency","Frequency","Monetary"]].min()
    all_max = rfm_c[["Recency","Frequency","Monetary"]].max()

    def norm(v, c):
        n = (v-all_min[c]) / (all_max[c]-all_min[c]+1e-9)
        return 1-n if c=="Recency" else n

    def pct(v, c):
        arr = rfm_c[c].values
        p = float((arr<=v).mean())*100
        return 100-p if c=="Recency" else p

    seg_avg = rfm_c[rfm_c["Segment"]==row["Segment"]][
        ["Recency","Frequency","Monetary"]].mean()
    cats = ["Recency Score","Frequency Score","Revenue Score"]
    cv   = [norm(row[c],c) for c in ["Recency","Frequency","Monetary"]]
    av   = [norm(seg_avg[c],c) for c in ["Recency","Frequency","Monetary"]]

    col_r,col_p = st.columns(2)
    with col_r:
        card("RFM Profile vs Segment Average",
             "Solid = this customer · Dashed = segment mean")
        try:
            fig_rad = go.Figure()
            fig_rad.add_trace(go.Scatterpolar(
                r=av+[av[0]], theta=cats+[cats[0]],
                fill="toself", name="Segment avg",
                line=dict(color="#334155", dash="dash", width=2),
                fillcolor=hex_to_rgba("#334155", 0.12),
            ))
            fig_rad.add_trace(go.Scatterpolar(
                r=cv+[cv[0]], theta=cats+[cats[0]],
                fill="toself", name=f"Customer {search_id}",
                line=dict(color=color, width=2.5),
                fillcolor=hex_to_rgba(color, 0.18),
            ))
            fig_rad.update_layout(
                polar=polar_layout(),
                paper_bgcolor=_PAPER_BG, font_color=_FONT_COL,
                font_family="Plus Jakarta Sans",
                legend=dict(font_size=10, bgcolor="rgba(0,0,0,0)",
                            font_color=_FONT_COL),
                margin=dict(l=10,r=10,t=10,b=10), height=340,
            )
            st.plotly_chart(fig_rad, use_container_width=True)
        except Exception as e:
            st.warning(f"Radar unavailable: {e}")
        end_card()

    with col_p:
        card("Percentile Rankings","Standing across all customers")
        for label,p in [
            ("📅 Recency Score",         pct(row["Recency"],  "Recency")),
            ("🔄 Frequency Rank",         pct(row["Frequency"],"Frequency")),
            (f"💰 Revenue Rank ({CODE})", pct(row["Monetary"], "Monetary")),
        ]:
            bc = "#22c55e" if p>=66 else ("#f59e0b" if p>=33 else "#ef4444")
            st.markdown(f"""
            <div class="pbar-row">
              <div class="pbar-head">
                <span class="pbar-lbl">{label}</span>
                <span class="pbar-val" style="color:{bc};">{p:.0f}th pct</span>
              </div>
              <div class="pbar-track">
                <div class="pbar-fill" style="width:{p:.0f}%;background:{bc};"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        seg_members = rfm_c[rfm_c["Segment"]==row["Segment"]]
        rev_rank    = int((rfm_c["Monetary"]<row["Monetary"]).sum())+1
        diff        = row["Monetary"] - seg_members["Monetary"].mean()
        st.markdown(f"""
        <div style="background:#071020;border:1px solid #162035;border-radius:9px;
                    padding:0.8rem 0.9rem;">
          <div style="color:#64748b;font-size:0.65rem;text-transform:uppercase;
                      letter-spacing:0.09em;margin-bottom:0.4rem;">Snapshot</div>
          <div style="font-size:0.79rem;color:#94a3b8;line-height:2.1;">
            Revenue rank: <b style="color:{color};">#{rev_rank}</b> of {len(rfm_c):,}<br>
            Segment size: <b style="color:{color};">{len(seg_members):,}</b> customers<br>
            Segment avg: <b style="color:{color};">{format_currency(seg_members['Monetary'].mean(),SYM)}</b><br>
            vs avg: <b style="color:{'#22c55e' if diff>=0 else '#ef4444'}">
              {'↑' if diff>=0 else '↓'} {format_currency(abs(diff),SYM)}</b>
          </div>
        </div>""", unsafe_allow_html=True)
        end_card()

    sec(f"📈 Revenue Distribution — {row['Segment']} Segment")
    card(f"Where Customer {search_id} Falls Among Peers",
         "Dashed line = this customer's revenue")
    seg_df = rfm_c[rfm_c["Segment"]==row["Segment"]].copy()
    fig_hist = px.histogram(
        seg_df, x="Monetary", nbins=25,
        color_discrete_sequence=[color],
        labels={"Monetary": f"Revenue ({CODE})"},
    )
    fig_hist.add_vline(
        x=row["Monetary"], line_color="#f8fafc", line_width=2, line_dash="dash",
        annotation_text=f"← {search_id}", annotation_font_color="#f8fafc",
    )
    apply(fig_hist, height=280, showlegend=False)
    fig_hist.update_xaxes(tickprefix=SYM)
    st.plotly_chart(fig_hist, use_container_width=True)
    end_card()

    sec("👥 Closest Peer Customers in Same Segment")
    peers = rfm_c[
        (rfm_c["Segment"]==row["Segment"]) &
        (rfm_c["customer_id"].astype(str)!=str(search_id))
    ].copy()
    peers["distance"] = (
        ((peers["Recency"]-row["Recency"])    /(all_max["Recency"]  -all_min["Recency"]  +1e-9))**2 +
        ((peers["Frequency"]-row["Frequency"])/(all_max["Frequency"]-all_min["Frequency"]+1e-9))**2 +
        ((peers["Monetary"]-row["Monetary"])  /(all_max["Monetary"] -all_min["Monetary"] +1e-9))**2
    )**0.5
    top10 = peers.nsmallest(10,"distance")[
        ["customer_id","Recency","Frequency","Monetary"]].copy()
    top10["Monetary"] = top10["Monetary"].apply(lambda v: format_currency(v, SYM))
    top10.rename(columns={"customer_id":"Customer ID","Recency":"Recency (days)",
                           "Monetary":f"Revenue ({CODE})"}, inplace=True)
    st.dataframe(top10.reset_index(drop=True), use_container_width=True)
    footer()


# ═════════════════════════════════════════════════════════════
# ROUTER
# ═════════════════════════════════════════════════════════════

if   page.startswith("🏠"): page_overview()
elif page.startswith("📂"): page_data_input()
elif page.startswith("🔢"): page_segmentation()
elif page.startswith("📊"): page_visualisation()
elif page.startswith("💡"): page_business_insights()
elif page.startswith("🔍"): page_customer_lookup()
