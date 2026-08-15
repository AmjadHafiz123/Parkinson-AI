from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import shap

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "shap"
    / "random_forest_model.joblib"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "parkinson_patient_features.csv"
)

SHAP_IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "shap"
    / "shap_feature_importance.csv"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Parkinson AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .hero {
        padding: 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a, #1e3a8a);
        color: white;
        margin-bottom: 2rem;
    }

    .hero h1 {
        font-size: 2.6rem;
        margin-bottom: 0.3rem;
    }

    .hero p {
        font-size: 1.1rem;
        opacity: 0.9;
    }

    .result-card {
        padding: 2rem;
        border-radius: 18px;
        background: white;
        border: 1px solid #e5e7eb;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }

    .healthy {
        color: #15803d;
        font-size: 2rem;
        font-weight: 700;
    }

    .parkinson {
        color: #dc2626;
        font-size: 2rem;
        font-weight: 700;
    }

    .metric-card {
        padding: 1.2rem;
        border-radius: 14px;
        background: white;
        border: 1px solid #e5e7eb;
        text-align: center;
    }

    .metric-title {
        color: #64748b;
        font-size: 0.9rem;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }

    .warning {
        padding: 1rem;
        border-radius: 12px;
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA / MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_shap_importance():
    if SHAP_IMPORTANCE_PATH.exists():
        return pd.read_csv(SHAP_IMPORTANCE_PATH)
    return None


model = load_model()
dataset = load_dataset()
shap_importance = load_shap_importance()

@st.cache_resource
def load_shap_explainer():
    return shap.TreeExplainer(model)

explainer = load_shap_explainer()

FEATURE_COLUMNS = [
    column
    for column in dataset.columns
    if column not in ["patient_id", "label"]
]


# ============================================================
# PREDICTION
# ============================================================

def predict_patient(patient_row):

    features = patient_row[FEATURE_COLUMNS]

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    # SHAP explanation for this patient
    shap_values = explainer.shap_values(features)

    # SHAP 0.52 + Random Forest gives:
    # (samples, features, classes)
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        patient_shap = shap_values[0, :, 1]

    elif isinstance(shap_values, list):
        patient_shap = shap_values[1][0]

    else:
        patient_shap = np.asarray(shap_values)[0]

    explanation = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "feature_value": features.iloc[0].values,
        "shap_value": patient_shap,
    })

    explanation["absolute_shap"] = (
        explanation["shap_value"].abs()
    )

    explanation = explanation.sort_values(
        "absolute_shap",
        ascending=False
    )

    return {
        "prediction": prediction,
        "healthy_probability": float(probabilities[0]),
        "parkinson_probability": float(probabilities[1]),
        "explanation": explanation,
    }

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🧠 Parkinson AI")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔬 Patient Assessment",
            "📊 Model Performance",
            "🔎 Feature Analysis",
            "ℹ️ About",
        ],
    )

    st.markdown("---")

    st.caption("Research prototype")
    st.caption("Not a medical diagnostic device")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>🧠 Parkinson AI</h1>

        <p>
        Explainable Parkinson's disease screening using
        smartwatch movement sensor data.
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.subheader("System Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Patients</div>
                <div class="metric-value">355</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Model Features</div>
                <div class="metric-value">112</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Random Forest Accuracy</div>
                <div class="metric-value">87.3%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">CV ROC-AUC</div>
                <div class="metric-value">77.0%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    left, right = st.columns([1.2, 1])

    with left:

        st.subheader("What does this system do?")

        st.write(
            """
            Parkinson AI analyses movement information collected from
            smartwatch sensors and uses machine learning to estimate
            whether a patient's movement profile is more consistent
            with the Healthy or Parkinson's class.
            """
        )

        st.write(
            """
            The system uses a patient-level Random Forest classifier
            together with SHAP explainability to show which sensor
            characteristics influenced the prediction.
            """
        )

    with right:

        st.subheader("Dataset")

        class_counts = dataset["label"].value_counts().sort_index()

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=["Healthy", "Parkinson's"],
                    values=[
                        int(class_counts.get(0, 0)),
                        int(class_counts.get(1, 0)),
                    ],
                    hole=0.55,
                )
            ]
        )

        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div class="warning">

        ⚠️ <b>Important:</b>
        This application is an academic/research prototype.
        It must not be used as a substitute for professional medical
        diagnosis or clinical assessment.

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PATIENT ASSESSMENT
# ============================================================

elif page == "🔬 Patient Assessment":

    st.subheader("Patient Assessment")

    st.write(
        "Select a patient from the processed dataset to test the "
        "complete inference pipeline."
    )

    patient_ids = dataset["patient_id"].tolist()

    selected_patient = st.selectbox(
        "Patient ID",
        patient_ids,
    )

    patient = dataset[
        dataset["patient_id"] == selected_patient
    ].iloc[[0]]

    if st.button(
        "🚀 Run Parkinson Assessment",
        type="primary",
        use_container_width=True,
    ):

        result = predict_patient(patient)

        st.markdown("### Assessment Result")

        result_col, probability_col = st.columns([1, 1])

        with result_col:

            if result["prediction"] == 1:

                st.markdown(
                    """
                    <div class="result-card">
                        <div class="parkinson">
                        🔴 Parkinson's Pattern Detected
                        </div>
                        <p>
                        The model predicts the Parkinson's class.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:

                st.markdown(
                    """
                    <div class="result-card">
                        <div class="healthy">
                        🟢 Healthy Pattern
                        </div>
                        <p>
                        The model predicts the Healthy class.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with probability_col:

            probability = result["parkinson_probability"]

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={
                        "suffix": "%",
                    },
                    title={
                        "text": "Parkinson's Probability"
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100]
                        },
                        "threshold": {
                            "line": {
                                "width": 4
                            },
                            "value": 50,
                        },
                    },
                )
            )

            fig.update_layout(
                height=280,
                margin=dict(
                    l=20,
                    r=20,
                    t=50,
                    b=20,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.markdown("### Prediction Probabilities")

        p1, p2 = st.columns(2)

        with p1:
            st.metric(
                "Healthy",
                f"{result['healthy_probability'] * 100:.1f}%",
            )

        with p2:
            st.metric(
                "Parkinson's",
                f"{result['parkinson_probability'] * 100:.1f}%",
            )

        st.markdown("### Patient Information")

        actual_label = int(patient["label"].iloc[0])

        st.write(
            {
                "Patient ID": selected_patient,
                "Actual class": (
                    "Parkinson's"
                    if actual_label == 1
                    else "Healthy"
                ),
                "Predicted class": (
                    "Parkinson's"
                    if result["prediction"] == 1
                    else "Healthy"
                ),
            }
        )
        st.markdown("### 🧠 Why did the model make this prediction?")
        explanation = result["explanation"]
        top_explanation = explanation.head(10)
        toward_parkinson = top_explanation[
        top_explanation["shap_value"] > 0
        ].copy()

        toward_healthy = top_explanation[
            top_explanation["shap_value"] < 0
        ].copy()
        chart_data = top_explanation.sort_values(
            "shap_value"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=chart_data["shap_value"],
                y=chart_data["feature"],
                orientation="h",
                text=chart_data["shap_value"].round(4),
                textposition="outside",
            )
        )

        fig.add_vline(
            x=0,
            line_width=2,
        )

        fig.update_layout(
            title="Patient-specific SHAP explanation",
            xaxis_title="Contribution toward Parkinson's prediction",
            yaxis_title="Feature",
            height=500,
            margin=dict(
                l=20,
                r=40,
                t=60,
                b=40,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.markdown("#### 🟢 Factors pushing the prediction toward Healthy")

        if len(toward_healthy) == 0:
            st.info("No strong Healthy-direction features were found.")
        else:
            for _, row in toward_healthy.iterrows():
                st.write(
                    f"**{row['feature']}** — "
                    f"value: `{row['feature_value']:.4f}` — "
                    f"contribution: `{row['shap_value']:.4f}`"
                )

        st.markdown(
            "#### 🔴 Factors pushing the prediction toward Parkinson's"
        )

        if len(toward_parkinson) == 0:
            st.info(
                "No strong Parkinson's-direction features were found."
            )
        else:
            for _, row in toward_parkinson.iterrows():
                st.write(
                    f"**{row['feature']}** — "
                    f"value: `{row['feature_value']:.4f}` — "
                    f"contribution: `{row['shap_value']:.4f}`"
                )
        st.markdown("#### 💡 Explanation")

        if result["prediction"] == 1:

            st.info(
                "The model predicted the Parkinson's class. "
                "The features with positive SHAP values contributed "
                "toward increasing the model's Parkinson's prediction, "
                "while negative SHAP values contributed in the opposite direction."
            )

        else:

            st.success(
                "The model predicted the Healthy class. "
                "Several of the strongest feature contributions moved "
                "the prediction away from the Parkinson's class."
            )

        st.caption(
            "SHAP values describe how features contributed to this "
            "machine-learning prediction. They are not medical reasoning "
            "and do not establish causation or diagnosis."
        )
# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.subheader("Model Performance")

    st.write(
        "Patient-level Random Forest evaluation."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Accuracy", "87.32%")

    with c2:
        st.metric("Balanced Accuracy", "78.52%")

    with c3:
        st.metric("F1 Score", "92.04%")

    with c4:
        st.metric("ROC-AUC", "86.25%")

    st.markdown("### Confusion Matrix")

    confusion = np.array(
        [
            [10, 6],
            [3, 52],
        ]
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=confusion,
            x=["Predicted Healthy", "Predicted Parkinson's"],
            y=["Actual Healthy", "Actual Parkinson's"],
            text=confusion,
            texttemplate="%{text}",
            hovertemplate="%{z}<extra></extra>",
        )
    )

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.markdown("### Cross-Validation Results")

    cv_data = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Balanced Accuracy",
                "Precision",
                "Recall",
                "F1",
                "ROC-AUC",
            ],
            "Mean": [
                0.7887,
                0.6615,
                0.8454,
                0.8912,
                0.8674,
                0.7699,
            ],
            "Std": [
                0.0345,
                0.0358,
                0.0194,
                0.0408,
                0.0242,
                0.0452,
            ],
        }
    )

    st.dataframe(
        cv_data,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FEATURE ANALYSIS
# ============================================================

elif page == "🔎 Feature Analysis":

    st.subheader("Explainable AI")

    st.write(
        """
        The model uses 112 patient-level features derived from
        smartwatch movement recordings.
        """
    )

    if shap_importance is not None:

        shap_data = shap_importance.head(20).copy()

        shap_data = shap_data.sort_values(
            "mean_abs_shap"
        )

        fig = go.Figure(
            go.Bar(
                x=shap_data["mean_abs_shap"],
                y=shap_data["feature"],
                orientation="h",
            )
        )

        fig.update_layout(
            title="Top 20 SHAP Features",
            height=650,
            xaxis_title="Mean |SHAP value|",
            yaxis_title="Feature",
            margin=dict(
                l=20,
                r=20,
                t=60,
                b=20,
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    else:

        st.info(
            "SHAP feature importance file was not found."
        )

    st.markdown("### Important Features")

    st.write(
        """
        Features such as signal variability, RMS, energy,
        range and median statistics appear among the most
        influential characteristics used by the model.
        """
    )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.subheader("About Parkinson AI")

    st.write(
        """
        Parkinson AI is an MSc research prototype for analysing
        smartwatch movement sensor data using machine learning.
        """
    )

    st.markdown("### Pipeline")

    st.code(
        """
Raw smartwatch recordings
        ↓
Sensor feature extraction
        ↓
Patient-level aggregation
        ↓
112 ML features
        ↓
Random Forest classifier
        ↓
Parkinson's probability
        ↓
SHAP explainability
        ↓
User-friendly result
        """,
        language="text",
    )

    st.markdown("### Dataset")

    st.write(
        """
        The current processed dataset contains 355 patients,
        including 79 Healthy and 276 Parkinson's patients.
        """
    )

    st.markdown("### Disclaimer")

    st.warning(
        """
        This software is intended for academic and research
        purposes only. It is not a medical device and should
        not be used to diagnose Parkinson's disease.
        """
    )
