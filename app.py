from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bankruptcy Prevention", page_icon="📉", layout="centered")

MODEL_PATH = Path(__file__).parent / "models" / "bankruptcy_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

LEVELS = {"Low": 0.0, "Medium": 0.5, "High": 1.0}

FEATURES = {
     "industrial_risk": "Industrial risk (High = riskier)",
    "management_risk": "Management risk (High = riskier)",
    "financial_flexibility": "Financial flexibility (High = stronger)",
    "credibility": "Credibility (High = stronger)",
    "competitiveness": "Competitiveness (High = stronger)",
    "operating_risk": "Operating risk (High = riskier)",
}

st.title("📉 Bankruptcy Prevention")
st.write("Rate the company on each factor to estimate its risk of bankruptcy.")

tab_predict, tab_about = st.tabs(["Predict", "About the model"])

with tab_predict:
    col1, col2 = st.columns(2)
    inputs = {}
    for i, (key, label) in enumerate(FEATURES.items()):
        with (col1 if i % 2 == 0 else col2):
            choice = st.select_slider(label, options=list(LEVELS), value="Medium", key=key)
            inputs[key] = LEVELS[choice]

    if st.button("Predict", type="primary", use_container_width=True):
        row = pd.DataFrame([inputs])[list(model.feature_names_in_)]
        prob = float(model.predict_proba(row)[0][1])

        st.divider()
        if prob >= 0.5:
            st.error(f"⚠️ High risk of bankruptcy ({prob:.0%})")
        else:
            st.success(f"✅ Low risk of bankruptcy ({prob:.0%})")

        st.progress(prob, text=f"Bankruptcy probability: {prob:.1%}")

        with st.expander("Inputs used"):
            st.dataframe(row, hide_index=True)

with tab_about:
    st.subheader("How it works")
    st.write(
        "The model is a Logistic Regression trained on 250 companies. "
        "Each factor is scored Low (0), Medium (0.5) or High (1)."
    )
    coef = pd.Series(model.coef_[0], index=model.feature_names_in_).sort_values()
    st.write("**Factor influence** (negative lowers bankruptcy risk, positive raises it):")
    st.bar_chart(coef)
    st.caption(
        "The dataset is small, so treat predictions as a decision aid rather than a verdict."
    )
