"""
Food-hygiene risk screener — Streamlit app.

Loads the trained model and scores a (prospective) food business from
characteristics known before any inspection: business type, region, and
urban/rural setting. Returns a calibrated risk estimate plus the factors
driving it.

Run locally:
    streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "model" / "model.joblib"
FIGDIR = Path(__file__).parent / "figures"

st.set_page_config(page_title="Food hygiene risk screener", page_icon="🍽️", layout="centered")


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["pipeline"], bundle["meta"]


def risk_band(prob: float, base: float) -> tuple[str, str]:
    if prob < base * 0.8:
        return "Lower than average", "#1d9e75"
    if prob < base * 1.5:
        return "Around average", "#ba7517"
    if prob < base * 3:
        return "Elevated", "#d85a30"
    return "High", "#c0392b"


def explain(pipeline, input_df) -> pd.DataFrame | None:
    """Per-prediction SHAP contributions for the chosen inputs (best effort)."""
    try:
        import shap
        pre, clf = pipeline.named_steps["pre"], pipeline.named_steps["clf"]
        Xt = pre.transform(input_df)
        names = [n.split("__", 1)[-1] for n in pre.get_feature_names_out()]
        sv = shap.TreeExplainer(clf).shap_values(Xt)
        vals = sv[1] if isinstance(sv, list) else sv
        if getattr(vals, "ndim", 2) == 3:
            vals = vals[:, :, 1]
        row = vals[0]
        active = [(names[i].replace("_", ": ", 1), float(row[i]))
                  for i in range(len(names)) if Xt[0][i] != 0]
        active.sort(key=lambda t: abs(t[1]), reverse=True)
        return pd.DataFrame(active, columns=["factor", "effect on risk"])
    except Exception:
        return None


pipeline, meta = load_model()
base = meta["base_rate"]

st.title("🍽️ Food hygiene risk screener")
st.caption(
    "Estimates the probability that a UK food business would receive a poor "
    "hygiene rating (FHRS 0–2), from characteristics known before any inspection. "
    "Trained on Food Standards Agency open data."
)

col1, col2 = st.columns(2)
with col1:
    business_type = st.selectbox("Business type", meta["business_types"])
    settlement_type = st.radio("Setting", meta["settlement_types"], horizontal=True)
with col2:
    region = st.selectbox("Region", meta["regions"])

input_df = pd.DataFrame([{
    "business_type": business_type,
    "region": region,
    "settlement_type": settlement_type,
}])

prob = float(pipeline.predict_proba(input_df)[0, 1])
band, colour = risk_band(prob, base)

st.divider()
left, right = st.columns([1, 1])
with left:
    st.metric("Estimated risk", f"{prob:.1%}", f"{prob - base:+.1%} vs average")
with right:
    st.markdown(
        f"<div style='padding:14px;border-radius:10px;background:{colour}22;"
        f"border:1px solid {colour}'>"
        f"<span style='color:{colour};font-weight:600;font-size:1.2rem'>{band}</span><br>"
        f"<span style='color:#888'>average business: {base:.1%}</span></div>",
        unsafe_allow_html=True,
    )

st.progress(min(prob / (base * 4), 1.0))

st.subheader("What's driving this estimate")
contrib = explain(pipeline, input_df)
if contrib is not None and not contrib.empty:
    contrib = contrib.set_index("factor")
    st.bar_chart(contrib, horizontal=True)
    st.caption("Positive bars push risk up; negative bars push it down (SHAP values).")
else:
    imp = pd.Series(meta["feature_importance"]).sort_values()
    st.bar_chart(imp)
    st.caption("Overall feature importance (per-prediction explanation unavailable).")

with st.expander("How good is this model — and what are its limits?"):
    m = meta["metrics"]
    a, b = st.columns(2)
    a.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    b.metric("PR-AUC", f"{m['pr_auc']:.3f}", f"base rate {base:.1%}")
    st.markdown(
        "- A **screening aid**, not a verdict. It ranks risk from *type and location* "
        "only — most of why any single business fails is specific to that business.\n"
        "- The data is **imbalanced** (~4.5% fail), so accuracy is misleading; ROC-AUC "
        "and precision-recall are the honest measures.\n"
        "- Inspection *recency* was deliberately **excluded** — it predicts well only "
        "through reverse causation (failing places get re-inspected sooner) and isn't "
        "known for a prospective business."
    )
    if (FIGDIR / "roc_curve.png").exists():
        st.image(str(FIGDIR / "roc_curve.png"))
    if (FIGDIR / "calibration.png").exists():
        st.image(str(FIGDIR / "calibration.png"))

st.caption("Data © Crown copyright, Food Standards Agency (FSA open API). "
           "Built by Abas Ukanga · for demonstration, not regulatory use.")
