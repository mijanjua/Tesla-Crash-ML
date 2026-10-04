"""Streamlit app: estimate whether Autopilot would be claimed for a fatal Tesla crash.

Run from the project root (train first so the model file exists):
    python -m src.train
    streamlit run app/streamlit_app.py
"""
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data  # noqa: E402
from src.train import MODEL_DIR  # noqa: E402

st.set_page_config(page_title="Tesla Autopilot Claim Predictor", page_icon="🚗", layout="wide")


@st.cache_resource
def load_model():
    path = MODEL_DIR / "autopilot_model.joblib"
    if not path.exists():
        return None, None
    return joblib.load(path), json.loads((MODEL_DIR / "metadata.json").read_text())


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
            threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05,
                                  help="Lower = catch more claimed cases, at the cost of more false alarms.")
            claimed = proba >= threshold
            st.metric("Probability Autopilot claimed", f"{proba:.0%}",
                      delta=f"{proba - meta['positive_rate']:+.0%} vs. dataset average ({meta['positive_rate']:.0%})",
                      delta_color="off")
            st.progress(proba)
            if claimed:
                st.markdown(f"### Likely **claimed** (≥ {threshold:.0%})")
            else:
                st.markdown(f"### Likely **not claimed** (< {threshold:.0%})")

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
        col.metric(f"CV {label}", f"{cv[f'{key} (mean)']:.2f}", help=f"± {cv[f'{key} (std)']:.2f} across 5 folds")
    st.caption("Baseline (always 'no'): ROC-AUC 0.50, PR-AUC ≈ positive rate. Recall/precision use a 0.5 threshold.")

    st.markdown("**Confusion matrix on the held-out test set** (model trained on the other 75%)")
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
    st.bar_chart(summary, x=group, y="claimed_rate", color="#2a78d6")
    st.dataframe(summary.style.format({"claimed_rate": "{:.0%}"}), width="stretch", hide_index=True)
