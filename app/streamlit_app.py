"""Streamlit app: estimate whether Autopilot would be claimed for a fatal Tesla crash.

Run from the project root (train first so the model file exists):
    python -m src.train
    streamlit run app/streamlit_app.py
"""
import base64
import json
import math
import sys
import warnings
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data  # noqa: E402
from src.train import MODEL_DIR, build_models  # noqa: E402

st.set_page_config(page_title="Tesla Autopilot Claim Predictor", page_icon="🚗", layout="wide")

TESLA_RED = "#e31937"

# Gauge ink per theme; st.html content isn't themed automatically
GAUGE_PALETTE = {
    "dark": {"ink": "#f2f2f0", "muted": "#a3a29b", "track": "rgba(227,25,55,0.22)",
             "hub": "#0e0e10", "pill_bg": "#2a2a30"},
    "light": {"ink": "#0b0b0b", "muted": "#6b6a65", "track": "rgba(227,25,55,0.13)",
              "hub": "#ffffff", "pill_bg": "#f0efec"},
}


def _gauge_point(fraction: float, radius: float, cx=150, cy=150) -> tuple[float, float]:
    """Point on the gauge's upper semicircle: fraction 0 = far left, 1 = far right."""
    angle = math.pi * (1 - fraction)
    return cx + radius * math.cos(angle), cy - radius * math.sin(angle)


def gauge_html(proba: float, threshold: float, base_rate: float, theme: str) -> str:
    """Speedometer-style gauge with a hero probability number and verdict pill underneath."""
    c = GAUGE_PALETTE.get(theme, GAUGE_PALETTE["dark"])
    claimed = proba >= threshold
    verdict = (f"⚠️ Likely claimed (≥ {threshold:.0%})" if claimed
               else f"✅ Likely not claimed (< {threshold:.0%})")
    pill = f"background:{TESLA_RED};color:#fff" if claimed else f"background:{c['pill_bg']};color:{c['ink']}"

    r = 120
    x0, y0 = _gauge_point(0, r)
    x1, y1 = _gauge_point(1, r)
    xv, yv = _gauge_point(proba, r)
    # A semicircle never spans more than 180°, so the large-arc flag is always 0
    value_arc = f"M {x0:.1f} {y0:.1f} A {r} {r} 0 0 1 {xv:.1f} {yv:.1f}" if proba > 0.002 else ""
    tx_in, ty_in = _gauge_point(threshold, r - 22)
    tx_out, ty_out = _gauge_point(threshold, r + 22)
    nx, ny = _gauge_point(proba, r - 34)

    ticks = ""
    for f in (0, 0.25, 0.5, 0.75, 1):
        lx, ly = _gauge_point(f, r + 36)
        ticks += (f'<text x="{lx:.1f}" y="{ly + 4:.1f}" text-anchor="middle" font-size="11" '
                  f'fill="{c["muted"]}">{f:.0%}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="-30 -26 360 204" font-family="sans-serif">
    <path d="M {x0:.1f} {y0:.1f} A {r} {r} 0 0 1 {x1:.1f} {y1:.1f}"
          fill="none" stroke="{c['track']}" stroke-width="22" stroke-linecap="round"/>
    <path d="{value_arc}" fill="none" stroke="{TESLA_RED}" stroke-width="22" stroke-linecap="round"/>
    <line x1="{tx_in:.1f}" y1="{ty_in:.1f}" x2="{tx_out:.1f}" y2="{ty_out:.1f}"
          stroke="{c['ink']}" stroke-width="2.5" stroke-linecap="round"/>
    {ticks}
    <line x1="150" y1="150" x2="{nx:.1f}" y2="{ny:.1f}" stroke="{c['ink']}" stroke-width="4" stroke-linecap="round"/>
    <circle cx="150" cy="150" r="9" fill="{c['ink']}"/>
    <circle cx="150" cy="150" r="3.5" fill="{c['hub']}"/>
  </svg>"""
    # st.html sanitizes away inline <svg>, so embed the gauge as an image
    svg_src = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()

    return f"""
<div style="font-family:'Source Sans Pro',sans-serif;text-align:center;max-width:340px;margin:0 auto">
  <img src="{svg_src}" style="width:100%;display:block"
       alt="Gauge: {proba:.0%} probability Autopilot claimed, threshold {threshold:.0%}"/>
  <div style="font-size:84px;font-weight:700;line-height:1;color:{c['ink']};letter-spacing:-2px;margin-top:4px">
    {proba:.0%}
  </div>
  <div style="font-size:14px;color:{c['muted']};margin-top:6px">probability Autopilot was claimed</div>
  <div style="display:inline-block;margin-top:14px;padding:6px 14px;border-radius:999px;font-weight:600;
              font-size:15px;{pill}">{verdict}</div>
  <div style="font-size:12px;color:{c['muted']};margin-top:10px">
    {proba - base_rate:+.0%} vs. dataset average ({base_rate:.0%}) · tick mark = decision threshold
  </div>
</div>
"""


@st.cache_resource
def load_model():
    path = MODEL_DIR / "autopilot_model.joblib"
    if not path.exists():
        return None, None
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    try:
        # A pickle from another scikit-learn version may load but behave wrongly, so treat it as a failure
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            model = joblib.load(path)
    except Exception:
        # The dataset is tiny, so retraining the same model here takes about a second
        df = data.load_dataset()
        model = build_models()[meta["model_name"]].fit(df[data.FEATURES], df[data.TARGET])
    return model, meta


@st.cache_data
def load_clean():
    return data.clean(data.load_raw())


model, meta = load_model()
clean = load_clean()

st.title("Tesla fatal crashes: was Autopilot claimed?")
st.caption(
    "Estimates the chance that Autopilot was *claimed* to be active in a fatal Tesla crash, "
    "based on when and where it happened, the car model, and who was killed. "
    "Trained on a small public dataset, so treat it as an educational demo, not evidence."
)

if model is None:
    st.error("No trained model found. Run `python -m src.train` from the project root first.")
    st.stop()

predict_tab, perf_tab, data_tab = st.tabs(["Predict", "Model performance", "Data explorer"])

with predict_tab:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("Crash details")
        c1, c2 = st.columns(2)
        crash_date = c1.date_input("Crash date", value=date(2022, 6, 1),
                                   min_value=date(2013, 1, 1), max_value=date(2030, 12, 31))
        tesla_model = c2.selectbox("Tesla model", ["S", "3", "X", "Y", "Unknown"], index=1)

        countries = sorted(c for c in clean["country"].unique() if c != "Unknown")
        country = c1.selectbox("Country", countries, index=countries.index("USA"))
        if country == "USA":
            states = sorted(s for s in clean.loc[clean["country"] == "USA", "state"].unique() if s != "-")
            state = c2.selectbox("State", states, index=states.index("CA"))
        else:
            state = "-"
            c2.text_input("State", value="n/a (outside USA)", disabled=True)

        st.markdown("**Who was killed**")
        v1, v2, v3, v4 = st.columns(4)
        tesla_driver = v1.number_input("Tesla driver", 0, 1, 1)
        tesla_occupant = v2.number_input("Tesla passengers", 0, 5, 0)
        other_vehicle = v3.number_input("Other vehicle", 0, 5, 0)
        cyclists_peds = v4.number_input("Cyclists / pedestrians", 0, 5, 0)

        description = st.text_input(
            "Short description (optional)",
            placeholder="e.g. Tesla hits parked truck on highway",
            help="Used only for keyword flags (pedestrian, motorcycle, fire, truck, fixed object, highway).",
        )

    deaths = tesla_driver + tesla_occupant + other_vehicle + cyclists_peds
    row = pd.DataFrame([{
        "year": crash_date.year, "month": crash_date.month, "country": country, "state": state,
        "model": tesla_model, "description": description, "deaths": deaths,
        "tesla_driver": tesla_driver, "tesla_occupant": tesla_occupant,
        "other_vehicle": other_vehicle, "cyclists_peds": cyclists_peds,
    }])
    features = data.add_features(row)[data.FEATURES]

    with right:
        st.subheader("Prediction")
        if deaths == 0:
            st.warning("Enter at least one death: every crash in this dataset was fatal.")
        else:
            proba = float(model.predict_proba(features)[0, 1])
            threshold = st.slider("Decision threshold", 0.05, 0.95, float(meta.get("threshold", 0.5)), 0.025,
                                  help="Defaults to the cut-off that maximised F1 in cross-validation. "
                                       "Lower = catch more claimed cases, at the cost of more false alarms.")
            theme = st.context.theme.type or "dark"
            st.html(gauge_html(proba, threshold, meta["positive_rate"], theme))

            with st.expander("Features sent to the model"):
                # Mixed text/number values, so show them as strings
                st.dataframe(features.T.astype(str).rename(columns={0: "value"}), width="stretch")

with perf_tab:
    st.subheader(f"Deployed model: {meta['model_name']}")
    st.write(
        f"Trained on {meta['n_rows']} crashes with a known label. "
        f"Only {meta['positive_rate']:.0%} have Autopilot claimed, so accuracy is misleading: "
        "always guessing 'no' would be ~87% accurate. The scores below focus on the claimed class."
    )
    cv = meta["cv_metrics"]
    cols = st.columns(4)
    for col, (key, label) in zip(cols, [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"),
                                        ("recall", "Recall"), ("precision", "Precision")]):
        col.metric(f"CV {label}", f"{cv[f'{key} (mean)']:.2f}",
                   help=f"± {cv[f'{key} (std)']:.2f} across 25 folds (5-fold CV repeated 5 times)")
    st.caption("Baseline (always 'no'): ROC-AUC 0.50, PR-AUC ≈ positive rate. CV recall/precision use a 0.5 threshold.")

    st.markdown(f"**Confusion matrix on the held-out test set** (model trained on the other 75%, "
                f"threshold {meta.get('threshold', 0.5):.2f})")
    cm = meta["test_metrics"]["confusion_matrix"]
    st.table(pd.DataFrame(cm, index=["Actual: not claimed", "Actual: claimed"],
                          columns=["Predicted: not claimed", "Predicted: claimed"]))

with data_tab:
    st.subheader("Autopilot-claimed rate in the data")
    labelled = clean[clean[data.TARGET].notna()].copy()
    labelled["region"] = labelled["country"].map(data.region_of)
    group = st.radio("Group by", ["year", "model", "region"], horizontal=True)
    summary = (labelled.groupby(group)[data.TARGET]
               .agg(claimed_rate="mean", crashes="size").reset_index())
    st.bar_chart(summary, x=group, y="claimed_rate", color=TESLA_RED)
    st.dataframe(summary.style.format({"claimed_rate": "{:.0%}"}), width="stretch", hide_index=True)
