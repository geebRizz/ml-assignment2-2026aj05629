# Streamlit app to compare the saved models on a test CSV.
# Models are trained in 2026aj05629.ipynb - this only loads them and predicts.
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "model"
BUNDLED_TEST_CSV = APP_DIR / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
TARGET = "diagnosis"
LABEL_MAP = {"M": 1, "B": 0, 1: 1, 0: 0}
CLASS_NAMES = ["Benign (0)", "Malignant (1)"]

st.set_page_config(page_title="Breast Cancer Classifier Comparison", page_icon="🩺", layout="wide")


@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / file) for name, file in MODEL_FILES.items()}


# returns X, y and an error message (None if the file is ok)
def prepare(df, feature_columns):
    if TARGET not in df.columns:
        return None, None, f"No '{TARGET}' column found in the CSV."
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        return None, None, f"Missing {len(missing)} feature column(s): {', '.join(missing[:5])}..."

    y = df[TARGET].map(LABEL_MAP)
    if y.isna().any():
        return None, None, f"'{TARGET}' should only have M/B or 1/0 values."
    X = df[feature_columns]  # same column order as training
    if X.isna().any().any():
        return None, None, "Some feature values are missing."
    return X, y.astype(int), None


def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_score) if y.nunique() == 2 else float("nan")
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": auc,
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return metrics, y_pred


def plot_confusion_matrix(y, y_pred, title):
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="RdPu", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                annot_kws={"size": 14})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


models = load_models()
feature_columns = list(next(iter(models.values())).feature_names_in_)

# sidebar
with st.sidebar:
    st.header("1. Test data")
    uploaded = st.file_uploader("Upload test CSV", type="csv",
                                help="Needs the 30 feature columns and 'diagnosis'.")
    use_bundled = False
    if uploaded is None and BUNDLED_TEST_CSV.exists():
        use_bundled = st.checkbox("Use test_data.csv from the repo", value=False)
        st.download_button("Download test_data.csv", BUNDLED_TEST_CSV.read_bytes(),
                           file_name="test_data.csv", mime="text/csv")

    st.header("2. Model")
    model_name = st.selectbox("Choose a model", list(MODEL_FILES))

    st.divider()
    st.caption("Models are loaded from `model/` (trained in the notebook).")

st.title("Breast Cancer Diagnosis - Model Comparison")
st.write("Breast Cancer Wisconsin (Diagnostic) dataset from UCI. "
         "Predicts malignant (1) or benign (0) from 30 cell nucleus features.")

if uploaded is not None:
    data = pd.read_csv(uploaded)
    source = uploaded.name
elif use_bundled:
    data = pd.read_csv(BUNDLED_TEST_CSV)
    source = "test_data.csv"
else:
    st.info("Upload a test CSV from the sidebar, or tick the checkbox to use test_data.csv.")
    st.stop()

X, y, error = prepare(data, feature_columns)
if error:
    st.error(error)
    st.stop()

st.success(f"Loaded {source}: {len(X)} rows ({int(y.sum())} malignant, {int((y == 0).sum())} benign)")

tab_model, tab_compare, tab_data = st.tabs(["Selected model", "Compare all models", "Data preview"])

with tab_model:
    metrics, y_pred = evaluate(models[model_name], X, y)
    st.subheader(f"{model_name} - metrics")
    for col, (label, value) in zip(st.columns(6), metrics.items()):
        col.metric(label, f"{value:.4f}")

    left, right = st.columns([1, 1.3])
    with left:
        st.markdown("**Confusion matrix**")
        st.pyplot(plot_confusion_matrix(y, y_pred, model_name))
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

with tab_compare:
    st.subheader("All models on this data")
    all_results = {name: evaluate(m, X, y)[0] for name, m in models.items()}
    table = pd.DataFrame(all_results).T
    table.index.name = "ML Model Name"
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#f7c6e0"),
                 width="stretch")

    metric = st.radio("Metric to plot", list(table.columns), horizontal=True, index=5)
    st.bar_chart(table[metric].sort_values(), horizontal=True, color="#c2185b")

with tab_data:
    st.dataframe(data.head(50), width="stretch")
    st.caption(f"Showing {min(50, len(data))} of {len(data)} rows.")
