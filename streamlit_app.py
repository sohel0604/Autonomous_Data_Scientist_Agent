# streamlit_app.py
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

from agents.coordinator import Coordinator
from utils.file_helpers import uploaded_file_to_df

load_dotenv()  # safe to keep even if .env is blank

st.set_page_config(page_title="Autonomous Data Scientist Agent (Offline)", layout="wide")
st.title("Autonomous Data Scientist Agent ")

st.markdown(
    """
Upload a CSV, choose the target column and press **Run Pipeline**.
This runs:
1. DataLoader Agent
2. EDA Agent
3. Model Trainer Agent (PyCaret)
4. Report Generator Agent (local transformers)
"""
)

uploaded_file = st.file_uploader("Upload CSV", accept_multiple_files=False, type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV to begin.")
    st.stop()

# read file safely
try:
    df = uploaded_file_to_df(uploaded_file)
except Exception as e:
    st.error(f"Failed to read uploaded CSV: {e}")
    st.stop()

st.subheader("Preview data")
st.dataframe(df.head(50))

# Choose target column
target = st.selectbox("Select target (label) column", options=list(df.columns))

# Run button
if st.button("Run Full Pipeline"):
    coordinator = Coordinator()  # offline default
    context = {"df": df, "target": target}
    with st.spinner("Running pipeline (this can take a few minutes)..."):
        try:
            result = coordinator.run_pipeline(context)
        except Exception as e:
            st.exception(e)
            st.stop()

    # EDA summary
    st.subheader("EDA Summary")
    eda = result.get("eda", {})
    st.write(f"Data shape: {eda.get('shape')}")
    st.write("Top missing counts:")
    st.write({k: v for k, v in list(eda.get("missing_values", {}).items())[:10]})
    if "target_balance" in eda:
        st.write("Target class balance:")
        st.write(eda["target_balance"])

    # AutoML compare
    st.subheader("AutoML results (PyCaret compare)")
    compare_df = result.get("automl", {}).get("compare_df")
    if compare_df is not None:
        st.dataframe(compare_df)

    # Model saved path
    model_path = result.get("automl", {}).get("model_path")
    if model_path and os.path.exists(model_path):
        st.success(f"Model saved at: `{model_path}`")
        with open(model_path, "rb") as f:
            st.download_button("Download model (.pkl)", data=f, file_name=os.path.basename(model_path))

    # AI explanation
    st.subheader("AI Explanation / Report")
    report = result.get("report")
    if report:
        st.markdown(report)
    else:
        st.write("No report generated.")
