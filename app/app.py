from pathlib import Path
import sys

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import shap
from src.inference.feature_builder import build_patient_features

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
        <p>Explainable Parkinson's disease screening using
        smartwatch movement sensor data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.subheader("Parkinson AI Dashboard")

    st.write(
        """
        An explainable machine-learning system that analyses
        smartwatch movement recordings and estimates whether
        the observed movement pattern is more consistent with
        the Healthy or Parkinson's class.
        """
    )

    # ========================================================
    # PROJECT OVERVIEW
    # ========================================================

    st.markdown("### 📌 Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">
                    Original PADS Participants
                </div>
                <div class="metric-value">
                    469
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">
                    Project Patients
                </div>
                <div class="metric-value">
                    355
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">
                    Model Features
                </div>
                <div class="metric-value">
                    112
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">
                    ML Model
                </div>
                <div class="metric-value">
                    Random Forest
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ========================================================
    # DATASET + SYSTEM
    # ========================================================

    left, right = st.columns([1.15, 1])

    with left:

        st.markdown("### 🧠 What Does Parkinson AI Do?")

        st.write(
            """
            Parkinson AI processes smartwatch movement recordings
            and converts the raw sensor signals into statistical
            features.
            """
        )

        st.write(
            """
            These features are aggregated at patient level and
            supplied to a Random Forest classifier. The model
            produces Healthy and Parkinson's probabilities.
            """
        )

        st.write(
            """
            SHAP explainability is then used to show which
            movement features contributed most strongly to the
            individual prediction.
            """
        )

        st.markdown("### 🔄 Prediction Pipeline")

        st.code(
            """
Smartwatch recordings
        ↓
Sensor feature extraction
        ↓
Recording-level features
        ↓
Patient-level aggregation
        ↓
112 ML features
        ↓
Random Forest
        ↓
Healthy / Parkinson's probability
        ↓
SHAP explanation
        ↓
User-friendly result
            """,
            language="text",
        )

    with right:

        st.markdown("### 📊 Project Dataset")

        class_counts = dataset["label"].value_counts().sort_index()

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=[
                        "Healthy",
                        "Parkinson's",
                    ],
                    values=[
                        int(class_counts.get(0, 0)),
                        int(class_counts.get(1, 0)),
                    ],
                    hole=0.55,
                    textinfo="label+percent",
                )
            ]
        )

        fig.update_layout(
            height=330,
            margin=dict(
                l=10,
                r=10,
                t=20,
                b=20,
            ),
            showlegend=True,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.caption(
            "This chart represents the 355-patient processed "
            "dataset used by this project."
        )

        st.link_button(
            "🔗 View Original PADS Dataset",
            "https://physionet.org/content/"
            "parkinsons-disease-smartwatch/1.0.0/",
        )

    # ========================================================
    # ORIGINAL DATASET INFORMATION
    # ========================================================

    st.markdown("### 📚 Original PADS Dataset")

    st.write(
        """
        The project is based on the PADS (Parkinsons Disease
        Smartwatch) dataset available through PhysioNet.
        """
    )

    dataset_col1, dataset_col2, dataset_col3, dataset_col4 = (
        st.columns(4)
    )

    with dataset_col1:
        st.metric(
            "Participants",
            "469",
        )

    with dataset_col2:
        st.metric(
            "Measurement Steps",
            "5,159",
        )

    with dataset_col3:
        st.metric(
            "Movement Tasks",
            "11",
        )

    with dataset_col4:
        st.metric(
            "Smartwatches",
            "2",
        )

    st.caption(
        "The original PADS dataset contains movement assessments "
        "recorded using two wrist-worn smartwatches. "
    )

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.markdown("### 🤖 Machine Learning Model")

    model_col1, model_col2, model_col3, model_col4 = st.columns(4)

    with model_col1:
        st.metric(
            "Algorithm",
            "Random Forest",
        )

    with model_col2:
        st.metric(
            "Input Features",
            "112",
        )

    with model_col3:
        st.metric(
            "Prediction",
            "2 Classes",
        )

    with model_col4:
        st.metric(
            "Explainability",
            "SHAP",
        )

    st.write(
        """
        The current model operates on patient-level features
        generated from smartwatch movement recordings.
        """
    )

    # ========================================================
    # NEW PATIENT WORKFLOW
    # ========================================================

    st.markdown("### 👤 New Patient Assessment")

    st.write(
        """
        A new user does not need to select an existing patient
        ID. The user can upload smartwatch movement recordings
        belonging to the person being assessed.
        """
    )

    step1, step2, step3, step4 = st.columns(4)

    with step1:
        st.markdown("### 1️⃣")
        st.markdown("**Upload**")
        st.caption(
            "Upload one or more smartwatch movement recordings."
        )

    with step2:
        st.markdown("### 2️⃣")
        st.markdown("**Extract**")
        st.caption(
            "Generate statistical features from the sensor signals."
        )

    with step3:
        st.markdown("### 3️⃣")
        st.markdown("**Predict**")
        st.caption(
            "The Random Forest estimates Healthy and Parkinson's probabilities."
        )

    with step4:
        st.markdown("### 4️⃣")
        st.markdown("**Explain**")
        st.caption(
            "SHAP identifies the features that influenced the prediction."
        )

    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.markdown("### 📈 Model Performance")

    perf1, perf2, perf3, perf4 = st.columns(4)

    with perf1:
        st.metric(
            "Accuracy",
            "87.32%",
        )

    with perf2:
        st.metric(
            "Balanced Accuracy",
            "78.52%",
        )

    with perf3:
        st.metric(
            "F1 Score",
            "92.04%",
        )

    with perf4:
        st.metric(
            "ROC-AUC",
            "86.25%",
        )

    st.caption(
        "These values represent the evaluation results currently "
        "reported by this project. See Model Performance for details."
    )

    # ========================================================
    # EXPLAINABLE AI
    # ========================================================

    st.markdown("### 🧠 Explainable AI")

    explain_col1, explain_col2 = st.columns(2)

    with explain_col1:

        st.info(
            """
            **Why this prediction?**

            SHAP identifies the individual features that pushed
            the model toward or away from the Parkinson's class.
            """
        )

    with explain_col2:

        st.warning(
            """
            **Important**

            SHAP explains the machine-learning model's behaviour.
            It does not represent medical reasoning or prove that
            a feature causes Parkinson's disease.
            """
        )

    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.markdown(
        """
        <div class="warning">

        ⚠️ <b>Research Prototype:</b>

        Parkinson AI is an MSc academic/research prototype.
        The prediction is generated from a machine-learning model
        trained on smartwatch movement data.

        This application is not a medical device and must not be
        used as a substitute for professional medical diagnosis,
        clinical assessment or treatment.

        </div>
        """,
        unsafe_allow_html=True,
    )
# ============================================================
# PATIENT ASSESSMENT
# ============================================================

elif page == "🔬 Patient Assessment":

    st.subheader("New Patient Assessment")

    st.write(
        """
        Upload smartwatch movement recordings for a new patient.
        The system will extract sensor features, aggregate them
        into the 112 features expected by the trained model, and
        generate a Parkinson's prediction with SHAP explainability.
        """
    )

    uploaded_files = st.file_uploader(
        "Upload smartwatch recordings",
        type=["txt"],
        accept_multiple_files=True,
        help=(
            "Upload the smartwatch movement recordings "
            "belonging to one patient."
        ),
    )

    if uploaded_files:

        st.success(
            f"✓ {len(uploaded_files)} smartwatch recordings uploaded"
        )

        if st.button(
            "🚀 Analyze New Patient",
            type="primary",
            use_container_width=True,
        ):

            import tempfile

            try:

                with st.spinner(
                    "Processing smartwatch recordings..."
                ):

                    with tempfile.TemporaryDirectory() as temp_dir:

                        temp_dir = Path(temp_dir)

                        recording_paths = []

                        for uploaded_file in uploaded_files:

                            file_path = (
                                temp_dir / uploaded_file.name
                            )

                            file_path.write_bytes(
                                uploaded_file.getbuffer()
                            )

                            recording_paths.append(file_path)

                        # ----------------------------------------
                        # Build 112 patient-level features
                        # ----------------------------------------

                        new_patient = build_patient_features(
                            recording_paths
                        )

                        # ----------------------------------------
                        # Validate feature structure
                        # ----------------------------------------

                        if (
                            new_patient.shape[1]
                            != len(FEATURE_COLUMNS)
                        ):
                            raise ValueError(
                                "The uploaded recordings produced "
                                f"{new_patient.shape[1]} features, "
                                f"but the model expects "
                                f"{len(FEATURE_COLUMNS)}."
                            )

                        if (
                            new_patient.columns.tolist()
                            != FEATURE_COLUMNS
                        ):
                            raise ValueError(
                                "The generated feature names or "
                                "feature order do not match the "
                                "trained model."
                            )

                        # ----------------------------------------
                        # Prediction + SHAP
                        # ----------------------------------------

                        result = predict_patient(
                            new_patient
                        )

                st.success("✓ Analysis complete")

                # =================================================
                # ASSESSMENT RESULT
                # =================================================

                st.markdown("### Assessment Result")

                result_col, probability_col = st.columns(
                    [1, 1]
                )

                with result_col:

                    if result["prediction"] == 1:

                        st.markdown(
                            """
                            <div class="result-card">
                                <div class="parkinson">
                                🔴 Parkinson's Pattern Detected
                                </div>
                                <p>
                                The model predicts the
                                Parkinson's class.
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
                                The model predicts the
                                Healthy class.
                                </p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                with probability_col:

                    probability = (
                        result["parkinson_probability"]
                    )

                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=probability * 100,
                            number={
                                "suffix": "%"
                            },
                            title={
                                "text":
                                "Parkinson's Probability"
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

                # =================================================
                # PROBABILITIES
                # =================================================

                st.markdown(
                    "### Prediction Probabilities"
                )

                p1, p2 = st.columns(2)

                with p1:

                    st.metric(
                        "Healthy",
                        (
                            f"{result['healthy_probability'] * 100:.1f}%"
                        ),
                    )

                with p2:

                    st.metric(
                        "Parkinson's",
                        (
                            f"{result['parkinson_probability'] * 100:.1f}%"
                        ),
                    )

                # =================================================
                # ASSESSMENT INFORMATION
                # =================================================

                st.markdown(
                    "### Assessment Information"
                )

                st.write(
                    {
                        "Recordings analyzed": len(
                            uploaded_files
                        ),
                        "Features generated": len(
                            FEATURE_COLUMNS
                        ),
                        "Predicted class": (
                            "Parkinson's"
                            if result["prediction"] == 1
                            else "Healthy"
                        ),
                    }
                )

                # =================================================
                # SHAP EXPLANATION
                # =================================================

                st.markdown(
                    "### 🧠 Why did the model make this prediction?"
                )

                explanation = result["explanation"]

                top_explanation = (
                    explanation.head(10)
                )

                toward_parkinson = (
                    top_explanation[
                        top_explanation["shap_value"] > 0
                    ].copy()
                )

                toward_healthy = (
                    top_explanation[
                        top_explanation["shap_value"] < 0
                    ].copy()
                )

                chart_data = (
                    top_explanation.sort_values(
                        "shap_value"
                    )
                )

                fig = go.Figure()

                fig.add_trace(
                    go.Bar(
                        x=chart_data["shap_value"],
                        y=chart_data["feature"],
                        orientation="h",
                        text=(
                            chart_data["shap_value"]
                            .round(4)
                        ),
                        textposition="outside",
                    )
                )

                fig.add_vline(
                    x=0,
                    line_width=2,
                )

                fig.update_layout(
                    title=(
                        "New Patient SHAP Explanation"
                    ),
                    xaxis_title=(
                        "Contribution toward "
                        "Parkinson's prediction"
                    ),
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

                # =================================================
                # HEALTHY FACTORS
                # =================================================

                st.markdown(
                    "#### 🟢 Factors pushing the prediction toward Healthy"
                )

                if len(toward_healthy) == 0:

                    st.info(
                        "No strong Healthy-direction "
                        "features were found."
                    )

                else:

                    for _, row in (
                        toward_healthy.iterrows()
                    ):

                        st.write(
                            f"**{row['feature']}** — "
                            f"value: "
                            f"`{row['feature_value']:.4f}` — "
                            f"contribution: "
                            f"`{row['shap_value']:.4f}`"
                        )

                # =================================================
                # PARKINSON FACTORS
                # =================================================

                st.markdown(
                    "#### 🔴 Factors pushing the prediction toward Parkinson's"
                )

                if len(toward_parkinson) == 0:

                    st.info(
                        "No strong Parkinson's-direction "
                        "features were found."
                    )

                else:

                    for _, row in (
                        toward_parkinson.iterrows()
                    ):

                        st.write(
                            f"**{row['feature']}** — "
                            f"value: "
                            f"`{row['feature_value']:.4f}` — "
                            f"contribution: "
                            f"`{row['shap_value']:.4f}`"
                        )

                # =================================================
                # EXPLANATION
                # =================================================

                st.markdown("#### 💡 Explanation")

                if result["prediction"] == 1:

                    st.info(
                        "The model predicted the "
                        "Parkinson's class. Features with "
                        "positive SHAP values contributed "
                        "toward increasing the model's "
                        "Parkinson's prediction, while "
                        "negative SHAP values contributed "
                        "in the opposite direction."
                    )

                else:

                    st.success(
                        "The model predicted the Healthy "
                        "class. Features with negative SHAP "
                        "values contributed toward moving "
                        "the prediction away from the "
                        "Parkinson's class."
                    )

                st.caption(
                    "SHAP values describe how features "
                    "contributed to this machine-learning "
                    "prediction. They are not medical "
                    "reasoning and do not establish "
                    "causation or diagnosis."
                )

            except Exception as error:

                st.error(
                    "Unable to analyze the uploaded "
                    "recordings."
                )

                st.exception(error)

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
        Parkinson AI is an MSc research prototype that uses
        smartwatch movement sensor data and machine learning
        to estimate whether a movement pattern is more
        consistent with the Healthy or Parkinson's class.
        """
    )

    # ---------------------------------------------------------
    # PROJECT WORKFLOW
    # ---------------------------------------------------------

    st.markdown("### 🔄 How the System Works")

    st.info(
        """
        The application follows a complete machine-learning
        pipeline from smartwatch recordings to an explainable
        prediction.
        """
    )

    st.markdown(
        """
        <div style="
            padding: 1.5rem;
            border-radius: 16px;
            background: white;
            border: 1px solid #e5e7eb;
            margin-bottom: 1rem;
        ">

        <h4>1️⃣ Smartwatch Movement Data</h4>
        <p>
        The system starts with movement recordings collected
        from smartwatch sensors. Each recording contains
        multiple sensor signals measured over time.
        </p>

        <hr>

        <h4>2️⃣ Sensor Feature Extraction</h4>
        <p>
        Each recording is converted into numerical features
        such as mean, standard deviation, minimum, maximum,
        median, range, RMS, energy and interquartile range.
        </p>

        <hr>

        <h4>3️⃣ Patient-Level Aggregation</h4>
        <p>
        Multiple recordings belonging to the same patient are
        aggregated to create a single patient-level representation.
        The current model uses <b>112 features</b>.
        </p>

        <hr>

        <h4>4️⃣ Random Forest Prediction</h4>
        <p>
        The trained Random Forest model receives the 112
        patient-level features and estimates the probability
        of the Healthy and Parkinson's classes.
        </p>

        <hr>

        <h4>5️⃣ SHAP Explainability</h4>
        <p>
        SHAP is used to identify which features contributed
        most strongly to the individual prediction.
        This provides an explanation of the model's decision.
        </p>

        <hr>

        <h4>6️⃣ User-Friendly Result</h4>
        <p>
        The application displays the predicted class,
        probability scores and the main features that pushed
        the prediction toward Healthy or Parkinson's.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # NEW PATIENT FLOW
    # ---------------------------------------------------------

    st.markdown("### 👤 New Patient Assessment")

    st.write(
        """
        A new user does not need to select an existing patient ID.
        Instead, the user uploads smartwatch movement recordings
        belonging to the person being assessed.
        """
    )

    new_patient_steps = [
        ("📁", "Upload Recordings",
         "Upload one or more smartwatch movement recordings for the person."),
        ("⚙️", "Build Features",
         "The application extracts statistical features from the recordings."),
        ("📐", "Create Patient Representation",
         "The recordings are aggregated into the 112 features required by the model."),
        ("🤖", "Run Random Forest",
         "The trained model calculates Healthy and Parkinson's probabilities."),
        ("🧠", "Generate SHAP Explanation",
         "SHAP identifies the features that contributed most to the prediction."),
        ("📊", "Display Result",
         "The UI shows the prediction, probabilities and explanation."),
    ]

    for icon, title, description in new_patient_steps:

        st.markdown(
            f"""
            <div style="
                display: flex;
                gap: 15px;
                align-items: flex-start;
                padding: 12px;
                margin: 7px 0;
                border-radius: 12px;
                background: #ffffff;
                border: 1px solid #e5e7eb;
            ">
                <div style="font-size: 1.5rem;">{icon}</div>
                <div>
                    <b>{title}</b>
                    <div style="color: #64748b; margin-top: 3px;">
                        {description}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------
    # MODEL
    # ---------------------------------------------------------

    st.markdown("### 🤖 Machine Learning Model")

    model_col1, model_col2, model_col3 = st.columns(3)

    with model_col1:
        st.metric("Model", "Random Forest")

    with model_col2:
        st.metric("Input Features", "112")

    with model_col3:
        st.metric("Explainability", "SHAP")

    st.write(
        """
        The model was trained using patient-level smartwatch
        movement features. The prediction is based on the
        learned statistical patterns in the training data.
        """

    )

   # ---------------------------------------------------------
   # DATASET
   # ---------------------------------------------------------

    st.markdown("### 📊 Dataset")

    dataset_col1, dataset_col2, dataset_col3 = st.columns(3)

    with dataset_col1:
        st.metric("Patients", "355")

    with dataset_col2:
        st.metric("Healthy", "79")

    with dataset_col3:
        st.metric("Parkinson's", "276")

    st.write(
        """
        This project uses the PADS (Parkinsons Disease Smartwatch)
        dataset provided through PhysioNet. The original dataset
        contains smartwatch movement recordings collected from
        participants performing neurologically designed movement
        tasks using wrist-worn smartwatches.
        """
    )

    st.markdown(
        """
        **Dataset source:**
        """
    )

    st.link_button(
        "🔗 Open PADS Dataset on PhysioNet",
        "https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/",
        use_container_width=False,
    )

    st.caption(
        "Varghese et al. (2024), PADS - Parkinsons Disease Smartwatch "
        "dataset, version 1.0.0, PhysioNet."
    )

    st.caption(
        "DOI: 10.13026/m0w9-zx22"
    )

    # ---------------------------------------------------------
    # EXPLAINABILITY
    # ---------------------------------------------------------

    st.markdown("### 🧠 What Does the Explanation Mean?")

    st.write(
        """
        SHAP values show how individual features influenced
        the model's prediction.
        """
    )

    st.markdown(
        """
        **Positive SHAP value**

        → pushes the model toward the Parkinson's class.

        **Negative SHAP value**

        → pushes the model away from the Parkinson's class
        and toward Healthy.

        **Larger absolute SHAP value**

        → stronger contribution to the model's prediction.
        """
    )

    st.warning(
        """
        SHAP explains the behaviour of the machine-learning
        model. It does not mean that a particular sensor
        feature medically causes Parkinson's disease.
        """
    )

    # ---------------------------------------------------------
    # PROJECT ARCHITECTURE
    # ---------------------------------------------------------

    st.markdown("### 🏗️ Project Architecture")

    st.code(
        """
Smartwatch Movement Recordings
              │
              ▼
     Feature Extraction
              │
              ▼
    Recording-Level Features
              │
              ▼
    Patient-Level Aggregation
              │
              ▼
       112 ML Features
              │
              ▼
     Random Forest Model
              │
        ┌─────┴─────┐
        ▼           ▼
   Prediction      SHAP
        │           │
        └─────┬─────┘
              ▼
      Streamlit User Interface
              │
              ▼
   Prediction + Probability
        + Explanation
        """,
        language="text",
    )

    # ---------------------------------------------------------
    # IMPORTANT
    # ---------------------------------------------------------

    st.markdown("### ⚠️ Important")

    st.warning(
        """
        Parkinson AI is an academic/research prototype.
        It is not a medical device and must not be used as
        a substitute for professional medical diagnosis,
        clinical assessment or treatment.
        """
    )