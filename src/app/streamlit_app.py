import sys
import os
import requests
API_URL = os.getenv("API_URL", "http://localhost:8000")

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
st.set_page_config(layout="wide")

from src.monitoring.drift import run_drift_report

st.title("Sentiment Analysis - Monitoring Dashboard")

st.header("Data Drift Check")

tab1, tab2 = st.tabs(["🔍 Drift Check", "✍️ Predict Review"])

with tab1:
    if st.button("Run Drift Check"):
        with st.spinner("Running drift analysis..."):
            report_path, message = run_drift_report()

        if report_path is None:
            st.warning(message)
        else:
            st.success(message)
            with open(report_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=4000, scrolling=True)

with tab2:
    st.subheader("Predict Sentiment")
    
    review_text = st.text_area("Enter a movie review:", height=150)

    if st.button("Predict"):
        st.session_state["predict_triggered"] = True
        st.session_state["review_input"] = review_text

    if st.session_state.get("predict_triggered"):
        text_to_predict = st.session_state.get("review_input", "").strip()

        if not text_to_predict:
            st.warning("Please enter a review before predicting.")
        else:
            with st.spinner("Getting prediction..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict",
                        json={"text": text_to_predict},
                        timeout=10,
                    )
                    if response.status_code == 200:
                        st.session_state["predict_result"] = response.json()
                        st.session_state["predict_error"] = None
                    else:
                        st.session_state["predict_result"] = None
                        st.session_state["predict_error"] = (
                            f"API returned status {response.status_code}: {response.text}"
                        )
                except requests.exceptions.ConnectionError:
                    st.session_state["predict_result"] = None
                    st.session_state["predict_error"] = (
                        "Could not connect to the API. Is it running?"
                    )
                except requests.exceptions.Timeout:
                    st.session_state["predict_result"] = None
                    st.session_state["predict_error"] = "The API took too long to respond."

        st.session_state["predict_triggered"] = False

    if st.session_state.get("predict_error"):
        st.error(st.session_state["predict_error"])

    if st.session_state.get("predict_result"):
        result = st.session_state["predict_result"]

        st.success(f"Prediction: **{result['label']}** (confidence: {result['confidence']:.2%})")

        st.write("Class probabilities:")
        st.bar_chart(result["scores"])

    