Yes. Below is the **complete `README.md` content**, ready to copy directly into your project's `README.md`.

````markdown
# 🧠 Parkinson AI

## Explainable Parkinson's Disease Screening Using Smartwatch Movement Data

Parkinson AI is an MSc research prototype that uses smartwatch movement sensor data, machine learning, and Explainable AI (XAI) to estimate whether a movement profile is more consistent with a Healthy or Parkinson's class.

The system provides an end-to-end pipeline:

```text
Smartwatch Movement Recordings
            ↓
      Feature Extraction
            ↓
   Patient-Level Aggregation
            ↓
      112 ML Features
            ↓
    Random Forest Model
            ↓
 Healthy / Parkinson's Probability
            ↓
       SHAP Explainability
            ↓
       Streamlit Web UI
````

> **⚠️ Important:** This project is an academic/research prototype. It is not a medical device and must not be used as a substitute for professional medical diagnosis or clinical assessment.

---

# 1. Problem Statement

Parkinson's disease is a neurological disorder that can affect movement, coordination, tremor, and motor control.

Movement information collected from wearable devices such as smartwatches can contain characteristics that may be useful for machine-learning research.

However, raw smartwatch sensor recordings are difficult to interpret directly.

This project addresses this problem by developing an AI-based system that can:

* Process smartwatch movement recordings.
* Extract statistical features from sensor signals.
* Aggregate multiple recordings at patient level.
* Use machine learning to classify movement patterns.
* Estimate Healthy and Parkinson's probabilities.
* Explain individual predictions using SHAP.
* Present the results through a user-friendly web interface.

The objective is not to replace a clinician, but to demonstrate an end-to-end machine-learning and explainable-AI workflow using wearable sensor data.

---

# 2. Use Case

Parkinson AI is designed as an academic and research prototype for studying the use of wearable movement data and machine learning.

## Example Use Case

A researcher or authorised user can upload smartwatch movement recordings belonging to a new participant.

The application processes the uploaded recordings and generates:

1. Movement features.
2. Patient-level features.
3. Machine-learning prediction.
4. Healthy probability.
5. Parkinson's probability.
6. SHAP-based feature contributions.
7. Visual explanation of the prediction.

Example:

```text
User uploads smartwatch recordings
              ↓
Application processes recordings
              ↓
112 patient-level features generated
              ↓
Random Forest prediction
              ↓
Healthy: 92.8%
Parkinson's: 7.2%
              ↓
SHAP explanation
              ↓
Top contributing movement features
```

The application is intended for:

* MSc academic work.
* Machine-learning research.
* Explainable AI demonstrations.
* Wearable-sensor research.
* Cloud and container deployment demonstrations.

It is not intended for:

* Medical diagnosis.
* Clinical decision-making.
* Treatment recommendations.
* Emergency medical assessment.

---

# 3. Solution Overview

The solution consists of multiple stages.

## 3.1 Raw Sensor Data

The project uses smartwatch movement recordings from the Parkinson's Disease Smartwatch Dataset available through PhysioNet.

The raw recordings contain time-series sensor measurements.

The project processes six sensor signal channels.

---

## 3.2 Feature Extraction

Statistical features are extracted from each sensor channel.

The current feature extraction process calculates:

* Mean
* Standard deviation
* Minimum
* Maximum
* Median
* Range
* Root Mean Square (RMS)
* Energy
* Interquartile Range (IQR)

For example:

```text
signal_1_mean
signal_1_std
signal_1_min
signal_1_max
signal_1_median
signal_1_range
signal_1_rms
signal_1_energy
signal_1_iqr
```

The same type of features is calculated for the six sensor channels.

---

## 3.3 Recording-Level Dataset

The feature extraction process produces a recording-level dataset:

```text
data/processed/parkinson_features.csv
```

The dataset contains information such as:

```text
patient_id
task
wrist
label
duration
num_samples
signal_1_*
signal_2_*
signal_3_*
signal_4_*
signal_5_*
signal_6_*
```

---

## 3.4 Patient-Level Aggregation

A patient may have multiple recordings from different tasks and wrist/device conditions.

Therefore, the project aggregates recording-level features by patient.

For numerical features, the system calculates:

```text
Mean across recordings
+
Standard deviation across recordings
```

The resulting patient-level dataset is:

```text
data/processed/parkinson_patient_features.csv
```

The trained Random Forest model expects:

```text
112 patient-level features
```

---

## 3.5 Machine Learning

A Random Forest classifier is used as the primary prediction model.

The model is stored as:

```text
data/processed/shap/random_forest_model.joblib
```

The model produces two class probabilities:

```text
Healthy probability
Parkinson's probability
```

The predicted class is determined from the model output.

---

## 3.6 Explainable AI

SHAP is used to explain the model's individual predictions.

The application uses:

```python
shap.TreeExplainer(model)
```

The explanation identifies which of the 112 features contributed most strongly to the prediction.

For example:

```text
signal_5_std
signal_5_energy_std
signal_5_rms_std
signal_5_rms_mean
signal_6_range_std
```

may appear among the strongest contributors for an individual prediction.

### SHAP Interpretation

For the Parkinson's-class explanation:

```text
Positive SHAP value
        ↓
Contribution toward Parkinson's class
```

and:

```text
Negative SHAP value
        ↓
Contribution away from Parkinson's class
```

Therefore, the UI can show:

```text
🟢 Factors pushing toward Healthy

🔴 Factors pushing toward Parkinson's
```

> SHAP values explain the behaviour of the machine-learning model. They do not represent medical reasoning, biological causation, or a clinical diagnosis.

---

# 4. Dataset

## Dataset Source

The project uses the:

**Parkinson's Disease Smartwatch Dataset**

provided through PhysioNet.

Official dataset:

[https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/)

Dataset version:

```text
1.0.0
```

Source:

```text
PhysioNet
```

---

## Dataset Description

The dataset contains smartwatch movement recordings collected for research involving Parkinson's disease.

The project processes movement time-series recordings and extracts statistical characteristics from six sensor channels.

The raw recordings are transformed into machine-learning features.

The overall transformation is:

```text
Raw smartwatch time-series
            ↓
Six sensor channels
            ↓
Statistical feature extraction
            ↓
Recording-level features
            ↓
Patient-level aggregation
            ↓
112 ML features
```

---

## Processed Dataset

The current processed patient-level dataset contains approximately:

```text
355 patients
112 model features
```

Current class distribution:

```text
Healthy       : 79
Parkinson's   : 276
```

The processed dataset is generated by the project's feature extraction and patient aggregation scripts.

---

# 5. AI/ML Approach

## 5.1 Model

The primary classification model is:

```text
Random Forest Classifier
```

Random Forest was selected because it is suitable for structured/tabular data and can model nonlinear relationships between features.

It also works well with tree-based SHAP explanations.

---

## 5.2 Input

The trained model expects:

```text
112 patient-level features
```

These features are generated from smartwatch movement recordings.

---

## 5.3 Output

The model produces:

```text
Predicted class
Healthy probability
Parkinson's probability
```

Example:

```text
Prediction: Healthy

Healthy probability: 92.8%
Parkinson's probability: 7.2%
```

---

## 5.4 Explainable AI

The project uses:

```text
SHAP
```

and:

```text
shap.TreeExplainer
```

to explain individual predictions.

The UI displays:

* Top contributing features.
* Feature values.
* SHAP values.
* Features contributing toward Healthy.
* Features contributing toward Parkinson's.
* SHAP contribution chart.

---

## 5.5 Machine Learning Pipeline

```text
Raw Recordings
       ↓
Feature Extraction
       ↓
Recording-Level Dataset
       ↓
Patient-Level Aggregation
       ↓
112 Features
       ↓
Random Forest
       ↓
Prediction
       ↓
SHAP Explanation
```

---

# 6. Application Architecture

The high-level application architecture is:

```text
                         ┌─────────────────────┐
                         │       User          │
                         │    / Researcher     │
                         └──────────┬──────────┘
                                    │
                                    │ Upload .txt
                                    ▼
                         ┌─────────────────────┐
                         │   Streamlit Web UI  │
                         │      app/app.py     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Feature Builder   │
                         │ build_patient_      │
                         │ features()          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Feature Extraction  │
                         │                     │
                         │ Mean                │
                         │ Std                 │
                         │ Min / Max           │
                         │ Median              │
                         │ Range               │
                         │ RMS                 │
                         │ Energy              │
                         │ IQR                 │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Patient-Level       │
                         │ Feature Vector      │
                         │                     │
                         │ 112 Features        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Random Forest       │
                         │ Classifier          │
                         └──────────┬──────────┘
                                    │
                       ┌────────────┴────────────┐
                       │                         │
                       ▼                         ▼
              ┌─────────────────┐       ┌─────────────────┐
              │ Prediction      │       │ SHAP            │
              │ Probability     │       │ TreeExplainer   │
              └────────┬────────┘       └────────┬────────┘
                       │                         │
                       └────────────┬────────────┘
                                    ▼
                         ┌─────────────────────┐
                         │   Result Dashboard  │
                         │                     │
                         │ Prediction          │
                         │ Probabilities       │
                         │ SHAP Chart          │
                         │ Important Features  │
                         └─────────────────────┘
```

---

# 7. Technology Stack

## Programming Language

```text
Python 3.13
```

## Data Processing

```text
Pandas
NumPy
```

## Machine Learning

```text
Scikit-learn
Random Forest
```

## Explainable AI

```text
SHAP
```

## Visualization

```text
Plotly
```

## Web Application

```text
Streamlit
```

## Model Serialization

```text
Joblib
```

## API

The project also contains a FastAPI application:

```text
src/api/main.py
```

FastAPI can be used to expose prediction functionality through HTTP endpoints.

## Containerization

```text
Docker
```

## Cloud Deployment

The application can be deployed to a container-compatible cloud platform such as:

```text
AWS
Azure
Google Cloud
Render
Railway
Streamlit Community Cloud
```

The exact platform depends on the cloud deployment configuration used for the assignment.

---

# 8. Project Structure

```text
Parkinson-AI/
│
├── app/
│   └── app.py
│
├── data/
│   ├── processed/
│   │   ├── parkinson_features.csv
│   │   ├── parkinson_patient_features.csv
│   │   │
│   │   └── shap/
│   │       ├── random_forest_model.joblib
│   │       └── shap_feature_importance.csv
│   │
│   └── pads-parkinsons-disease-smartwatch-dataset-1.0.0/
│
├── src/
│   │
│   ├── api/
│   │   └── main.py
│   │
│   ├── inference/
│   │   ├── predictor.py
│   │   └── feature_builder.py
│   │
│   ├── analyze_dataset.py
│   ├── create_patient_dataset.py
│   ├── explain_shap.py
│   ├── extract_features.py
│   ├── inspect_dataset.py
│   ├── train_baseline.py
│   ├── train_cross_validation.py
│   ├── train_patient_baseline.py
│   ├── train_random_forest.py
│   └── train_xgboost.py
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# 9. Local Setup Instructions

## 9.1 Prerequisites

Install:

* Python
* Git
* Docker (optional)

Recommended Python version:

```text
Python 3.13
```

---

## 9.2 Clone the Repository

```powershell
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project directory:

```powershell
cd Parkinson-AI
```

---

## 9.3 Create Virtual Environment

```powershell
python -m venv .venv
```

---

## 9.4 Activate Virtual Environment

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

Then:

```powershell
.venv\Scripts\Activate.ps1
```

---

## 9.5 Install Dependencies

```powershell
pip install -r requirements.txt
```

---

# 10. Data Processing

If the raw dataset is available locally, the feature extraction pipeline can be executed.

## Feature Extraction

Run:

```powershell
python src/extract_features.py
```

This generates:

```text
data/processed/parkinson_features.csv
```

The recording-level dataset contains:

```text
patient_id
task
wrist
label
duration
num_samples
signal features
```

---

## Patient-Level Dataset

Run:

```powershell
python src/create_patient_dataset.py
```

This generates:

```text
data/processed/parkinson_patient_features.csv
```

The resulting dataset contains the patient-level feature representation used by the model.

---

# 11. Model Training

The project contains multiple model training scripts.

For the Random Forest model:

```powershell
python src/train_random_forest.py
```

The trained model should be saved as:

```text
data/processed/shap/random_forest_model.joblib
```

The project also contains scripts for:

```text
Baseline model
Patient-level baseline
Cross-validation
Random Forest
XGBoost
SHAP analysis
```

---

# 12. Inference Testing

The inference pipeline can be tested using:

```powershell
python src/inference/predictor.py
```

A successful inference test should report:

```text
Model loaded
Expected features: 112
Dataset loaded
Features found: 112
```

and then display:

```text
Patient ID
Actual label
Prediction
Healthy probability
Parkinson's probability
Top contributing features
```

Example:

```text
Prediction: Healthy

Healthy probability: 92.80%
Parkinson's probability: 7.20%
```

---

# 13. Running the Web Application

Start Streamlit:

```powershell
streamlit run app/app.py
```

The terminal will display a local address similar to:

```text
http://localhost:8501
```

Open the address in a web browser.

---

# 14. Web Application Usage

The Streamlit application contains the following pages:

```text
🏠 Dashboard
🔬 Patient Assessment
📊 Model Performance
🔎 Feature Analysis
ℹ️ About
```

---

## 14.1 Dashboard

The Dashboard provides a high-level overview of the system.

It displays:

* Number of patients.
* Number of model features.
* Model performance.
* Dataset class distribution.
* Project description.
* Research disclaimer.

The Dashboard explains that the system:

```text
Processes smartwatch movement data
        ↓
Extracts features
        ↓
Uses Random Forest
        ↓
Produces probabilities
        ↓
Uses SHAP for explanation
```

---

# 15. New Patient Assessment

The Patient Assessment page is designed for a **new patient**.

The user does not select a patient ID.

Instead, the user uploads movement recordings.

## Step 1

Open:

```text
🔬 Patient Assessment
```

## Step 2

Upload smartwatch recording files.

Supported format:

```text
.txt
```

Multiple files can be uploaded.

The uploaded files should belong to the same patient/participant.

---

## Step 3

Click:

```text
🚀 Analyze New Patient
```

---

## Step 4

The system processes the recordings.

Internally:

```text
Uploaded .txt files
        ↓
Temporary processing directory
        ↓
Feature extraction
        ↓
Feature aggregation
        ↓
112 features
```

---

## Step 5

The Random Forest model performs prediction.

The application displays:

```text
Prediction
Healthy probability
Parkinson's probability
```

Example:

```text
Prediction:
Healthy

Healthy:
92.8%

Parkinson's:
7.2%
```

---

# 16. Prediction Explanation

The application provides an explanation section:

```text
🧠 Why did the model make this prediction?
```

The system calculates SHAP values for the patient's 112 features.

The UI displays a SHAP bar chart.

Conceptually:

```text
                Parkinson's direction
                        ↑
                        │
signal_5_std       █████████
signal_6_range     ███████
signal_5_rms       █████
                        │
------------------------0----------------
                        │
signal_1_iqr       ███████
signal_5_energy    █████████
                        │
                        ↓
                  Healthy direction
```

The application also separates the features into:

```text
🟢 Factors pushing the prediction toward Healthy

🔴 Factors pushing the prediction toward Parkinson's
```

Each feature can show:

```text
Feature name
Feature value
SHAP contribution
```

---

# 17. Model Performance

The Model Performance page displays the current experimental model results.

Current reported metrics:

| Metric            |  Value |
| ----------------- | -----: |
| Accuracy          | 87.32% |
| Balanced Accuracy | 78.52% |
| F1 Score          | 92.04% |
| ROC-AUC           | 86.25% |

These metrics are displayed for research and demonstration purposes.

They should not be interpreted as clinical performance.

---

# 18. Cross-Validation

Current cross-validation results:

| Metric            |   Mean | Standard Deviation |
| ----------------- | -----: | -----------------: |
| Accuracy          | 0.7887 |             0.0345 |
| Balanced Accuracy | 0.6615 |             0.0358 |
| Precision         | 0.8454 |             0.0194 |
| Recall            | 0.8912 |             0.0408 |
| F1                | 0.8674 |             0.0242 |
| ROC-AUC           | 0.7699 |             0.0452 |

The cross-validation results provide an additional estimate of model performance across different validation splits.

---

# 19. Feature Analysis

The Feature Analysis page provides global model explainability.

It uses the generated SHAP feature importance data:

```text
data/processed/shap/shap_feature_importance.csv
```

The UI displays the most influential features.

Examples may include:

```text
signal_5_std_std
signal_5_energy_std
signal_5_rms_std
signal_5_rms_mean
signal_6_range_std
signal_5_min_std
signal_6_median_std
```

The global feature analysis answers:

> Which movement features are generally important to the trained model?

This is different from individual SHAP analysis.

### Global Explanation

```text
Feature Analysis
        ↓
What features are important overall?
```

### Individual Explanation

```text
Patient Assessment
        ↓
Why did the model make this particular prediction?
```

---

# 20. About Page

The About page describes the complete project pipeline:

```text
Raw smartwatch recordings
        ↓
Sensor feature extraction
        ↓
Recording-level features
        ↓
Patient-level aggregation
        ↓
112 ML features
        ↓
Random Forest classifier
        ↓
Healthy / Parkinson's probability
        ↓
SHAP explainability
        ↓
Streamlit web application
```

It also provides information about:

* Dataset.
* Machine-learning approach.
* Explainable AI.
* Project purpose.
* Limitations.
* Research disclaimer.

---

# 21. API

The project contains a FastAPI application:

```text
src/api/main.py
```

The API component is intended to provide programmatic access to prediction functionality.

The API architecture is:

```text
Client
   ↓
FastAPI
   ↓
Feature Builder
   ↓
Random Forest
   ↓
Prediction
   ↓
JSON Response
```

To inspect the API implementation:

```powershell
Get-Content src/api/main.py
```

The exact API endpoints depend on the implementation in `src/api/main.py`.

---

# 22. Docker Instructions

Docker can be used to package the application and its dependencies into a reproducible environment.

---

## 22.1 Build Docker Image

From the project root:

```powershell
docker build -t parkinson-ai .
```

---

## 22.2 Run Docker Container

```powershell
docker run -p 8501:8501 parkinson-ai
```

Then open:

```text
http://localhost:8501
```

---

## 22.3 Run in Background

```powershell
docker run -d -p 8501:8501 --name parkinson-ai parkinson-ai
```

---

## 22.4 Check Running Container

```powershell
docker ps
```

---

## 22.5 Stop Container

```powershell
docker stop parkinson-ai
```

---

## 22.6 Remove Container

```powershell
docker rm parkinson-ai
```

---

# 23. Cloud Deployment

The application can be deployed as a Dockerized Streamlit application to a cloud platform.

Possible deployment architecture:

```text
                     Cloud Platform
                           │
                           ▼
                    Docker Container
                           │
                           ▼
                    Streamlit Server
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
        ML Model                    SHAP
              │                         │
              └────────────┬────────────┘
                           ▼
                     Web Interface
```

Possible cloud platforms include:

* AWS
* Microsoft Azure
* Google Cloud
* Render
* Railway
* Streamlit Community Cloud

The final deployment platform should be documented according to the cloud service actually used for the assignment.

---

# 24. Deployment Requirements

The deployed application requires:

```text
Python environment
        +
Required Python packages
        +
Trained Random Forest model
        +
Feature-building code
        +
Streamlit application
```

The model file must be available at:

```text
data/processed/shap/random_forest_model.joblib
```

The SHAP feature importance file should be available at:

```text
data/processed/shap/shap_feature_importance.csv
```

if global feature analysis is required.

---

# 25. End-to-End Project Workflow

The complete project workflow is:

```text
                    PHYSIONET
                       │
                       ▼
        Parkinson's Smartwatch Dataset
                       │
                       ▼
             Raw Sensor Recordings
                       │
                       ▼
              Feature Extraction
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
      Six Sensor                Statistical
       Channels                  Features
          │                         │
          └────────────┬────────────┘
                       ▼
             Recording-Level Data
                       │
                       ▼
             Patient-Level Grouping
                       │
                       ▼
                 Mean + Std
                       │
                       ▼
                 112 Features
                       │
                       ▼
              Random Forest Model
                       │
                       ▼
                Model Prediction
                       │
              ┌────────┴─────────┐
              │                  │
              ▼                  ▼
        Probability          SHAP Values
              │                  │
              └────────┬─────────┘
                       ▼
                 Streamlit UI
                       │
                       ▼
              New Patient Upload
                       │
                       ▼
                 Final Result
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
       Prediction              Explanation
          │                         │
          ▼                         ▼
    Healthy/Parkinson's        SHAP Features
```

---

# 26. Responsible AI

The project incorporates basic responsible-AI principles.

## Explainability

The system provides SHAP explanations instead of only displaying a prediction.

## Transparency

The application shows:

* Model type.
* Feature count.
* Probabilities.
* Model performance.
* Important features.

## Limitations

The application clearly communicates that the model is a research prototype.

## Human Oversight

Predictions should not be treated as medical decisions.

Professional clinical assessment remains necessary for real-world medical use.

---

# 27. Limitations

The project has several limitations.

### 1. Research Prototype

The system has been developed as an MSc academic project.

### 2. No Clinical Validation

The model has not been clinically validated.

### 3. Dataset Limitations

Machine-learning performance depends on the characteristics and distribution of the dataset.

### 4. Generalisation

The model may not generalise to:

* Different smartwatch devices.
* Different sensor configurations.
* Different populations.
* Different recording environments.
* Different clinical settings.

### 5. Class Imbalance

The processed dataset contains more Parkinson's examples than Healthy examples.

Therefore, accuracy alone should not be used to judge model quality.

### 6. Explainability Limitations

SHAP explains model behaviour.

It does not prove that a feature causes Parkinson's disease or that the feature represents a clinical biomarker.

### 7. No Medical Diagnosis

The output is a machine-learning classification and should not be interpreted as a diagnosis.

---

# 28. Example Prediction

An example prediction may look like:

```text
========================================
PARKINSON AI ASSESSMENT
========================================

Prediction:
Healthy

Healthy Probability:
92.8%

Parkinson's Probability:
7.2%

========================================
TOP CONTRIBUTING FEATURES
========================================

signal_5_std
Contribution: -0.0274

signal_5_energy_std
Contribution: -0.0238

signal_5_rms_std
Contribution: -0.0229

signal_5_rms_mean
Contribution: -0.0229

signal_6_range_std
Contribution: -0.0223
```

The web interface provides the same information in a visual format.

---

# 29. Research Interpretation

The project demonstrates the integration of:

```text
Wearable Sensor Data
        +
Feature Engineering
        +
Patient-Level Machine Learning
        +
Random Forest
        +
SHAP Explainability
        +
Streamlit
        +
FastAPI
        +
Docker
        +
Cloud Deployment
```

The main research contribution is an end-to-end explainable machine-learning workflow for smartwatch movement data.

---

# 30. Dataset Citation

The project uses the Parkinson's Disease Smartwatch Dataset hosted by PhysioNet.

Official dataset page:

[https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/)

Users of the dataset should follow the citation, licensing, and usage requirements specified by PhysioNet and the dataset authors.

---

# 31. Future Improvements

Potential future improvements include:

* Larger and more diverse datasets.
* Additional smartwatch sensor modalities.
* Improved class balancing.
* Hyperparameter optimisation.
* External validation.
* More robust cross-validation.
* Calibration of prediction probabilities.
* More advanced time-series models.
* Deep-learning approaches.
* Model monitoring after deployment.
* Automated cloud deployment.
* Authentication and user management.
* Secure storage of uploaded recordings.
* Prediction audit logs.
* More detailed SHAP visualisations.
* REST API integration with the Streamlit frontend.

---

# 32. Summary

Parkinson AI provides an end-to-end explainable machine-learning system for smartwatch movement analysis.

The system:

```text
1. Receives smartwatch recordings
2. Extracts movement features
3. Aggregates features at patient level
4. Generates 112 machine-learning features
5. Uses a Random Forest classifier
6. Produces Healthy/Parkinson's probabilities
7. Uses SHAP to explain predictions
8. Displays results through Streamlit
9. Provides API functionality through FastAPI
10. Supports Docker-based deployment
```

The final application demonstrates how wearable sensor data, machine learning, Explainable AI, web technologies, APIs, containers, and cloud deployment can be combined into a single research prototype.

---

# ⚠️ Medical Disclaimer

**Parkinson AI is an academic/research prototype and is not a medical device.**

The predictions generated by this application are machine-learning outputs based on smartwatch movement data.

They:

* Do not constitute a medical diagnosis.
* Should not be used to make clinical decisions.
* Should not replace professional medical assessment.
* Do not establish the presence or absence of Parkinson's disease.

For real-world medical concerns, users should consult a qualified healthcare professional.

---

# 👨‍🎓 MSc Project

**Project:** Parkinson AI

**Purpose:** MSc Cloud Computing / Artificial Intelligence Research Project

**Core Technologies:**

```text
Python
Pandas
NumPy
Scikit-learn
Random Forest
SHAP
Plotly
Streamlit
FastAPI
Joblib
Docker
```

**Dataset:**

Parkinson's Disease Smartwatch Dataset

**Dataset Provider:**

PhysioNet

**Dataset URL:**

[https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/)

````

### One thing I recommend before you submit

There is one **important consistency issue** in your current project: your README says **355 patients / 79 Healthy / 276 Parkinson's**, while some of your UI metrics and model-performance values appear to come from different experiments. For an MSc submission, the README, Dashboard, Model Performance page, and actual training output should all use the **same final experiment's numbers**.

Also, don't leave placeholders like:

```text
<YOUR_GITHUB_REPOSITORY_URL>
````

or claim a cloud provider you haven't actually deployed to. Replace those with the actual deployment details before submission.
