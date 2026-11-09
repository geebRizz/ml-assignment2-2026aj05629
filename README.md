# Breast Cancer Classification - ML Assignment 2

M.Tech AIML, BITS Pilani WILP

- GitHub repo: <https://github.com/geebRizz/ml-assignment2-2026aj05629>
- Streamlit app: <https://ml-assignment2-2026aj05629-6ypu2uqcpxg8chw5sggyyh.streamlit.app/>

## a. Problem statement

Predict whether a breast tumour is malignant or benign using measurements of cell nuclei from a fine-needle aspirate (FNA) image. This is a binary classification problem.

I trained five classifiers on the same split, compared them on six metrics, and put them in a Streamlit app where you can upload a test CSV and check how each model does on it.

## b. Dataset description

Breast Cancer Wisconsin (Diagnostic), UCI ML Repository (id 17):
<https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>

| Property | Value |
|---|---|
| Instances | 569 |
| Features | 30, all numeric |
| Target | `diagnosis` - M = 1 (malignant), B = 0 (benign) |
| Classes | 357 benign (62.7%), 212 malignant (37.3%) |
| Missing values | None |

Each nucleus has 10 base measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. For each of these the dataset gives the mean, the standard error (`_se`) and the "worst" value (mean of the 3 largest), so 30 features in total, e.g. `radius_mean`, `radius_se`, `radius_worst`.

Preprocessing:
- Dropped the `id` column.
- Mapped `diagnosis` to 1 (M) and 0 (B).
- 80/20 stratified split with `random_state=29` (455 train, 114 test). The test part is saved as `test_data.csv`.
- Logistic Regression, KNN and SVM use a `StandardScaler` inside their pipeline, fitted on the training data only. Decision Tree and Naive Bayes use the raw features.

## c. GitHub repository

<https://github.com/geebRizz/ml-assignment2-2026aj05629>

```
├── app.py                     # Streamlit app
├── 2026aj05629.ipynb          # training and evaluation notebook
├── requirements.txt
├── README.md
├── test_data.csv              # test split (30 features + diagnosis)
├── data/
│   └── breast_cancer_wisconsin.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── model_comparison.csv
```

To run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## d. Models used

| # | Model | Configuration |
|---|---|---|
| 1 | Logistic Regression | `StandardScaler` + `LogisticRegression(max_iter=5000)` |
| 2 | Decision Tree | `DecisionTreeClassifier(max_depth=5)` |
| 3 | K-Nearest Neighbors | `StandardScaler` + `KNeighborsClassifier(n_neighbors=7)` |
| 4 | Naive Bayes | `GaussianNB()` |
| 5 | Support Vector Machine | `StandardScaler` + `SVC(kernel="rbf", probability=True)` |

### Comparison table (test set, 114 rows)

Malignant (1) is the positive class for precision, recall and F1. AUC uses the predicted probability of class 1.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Decision Tree | 0.9649 | 0.9722 | 0.9130 | 1.0000 | 0.9545 | 0.9286 |
| KNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Naive Bayes | 0.9474 | 0.9873 | 0.9286 | 0.9286 | 0.9286 | 0.8869 |
| Support Vector Machine | 0.9825 | 0.9997 | 0.9762 | 0.9762 | 0.9762 | 0.9623 |

Confusion matrix counts on the test set (72 benign, 42 malignant):

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Logistic Regression | 72 | 0 | 0 | 42 |
| Decision Tree | 68 | 4 | 0 | 42 |
| KNN | 72 | 0 | 0 | 42 |
| Naive Bayes | 69 | 3 | 3 | 39 |
| Support Vector Machine | 71 | 1 | 1 | 41 |

Since two models got a perfect score on the test set, I also ran 5-fold stratified cross-validation on the training data (mean ± std) to get a better idea of how they compare:

| ML Model Name | Accuracy | AUC | F1 | MCC |
|---|---|---|---|---|
| Logistic Regression | 0.965 ± 0.018 | 0.990 ± 0.009 | 0.952 ± 0.024 | 0.925 ± 0.038 |
| Decision Tree | 0.925 ± 0.034 | 0.895 ± 0.032 | 0.896 ± 0.046 | 0.841 ± 0.074 |
| KNN | 0.965 ± 0.025 | 0.987 ± 0.018 | 0.951 ± 0.035 | 0.925 ± 0.055 |
| Naive Bayes | 0.930 ± 0.013 | 0.983 ± 0.009 | 0.903 ± 0.018 | 0.849 ± 0.028 |
| Support Vector Machine | 0.969 ± 0.026 | 0.992 ± 0.011 | 0.959 ± 0.035 | 0.934 ± 0.056 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Got all 114 test rows right. CV accuracy is 0.965, so the test split is probably on the easy side, but it's still one of the best models and fairly consistent across folds (± 0.018). After scaling the two classes are almost linearly separable, so a linear model does well here. |
| Decision Tree | Weakest model overall. It didn't miss any malignant case (recall 1.0) but had 4 false positives, so precision drops to 0.913. AUC is much lower than the rest (0.895 in CV) because a depth-5 tree only gives a handful of distinct probability values. It also has the biggest spread across folds, so it's overfitting somewhat. |
| KNN | Also perfect on the test set and the same CV accuracy as Logistic Regression (0.965), but with more variation between folds. Scaling is important here, otherwise large features like `area_mean` dominate the distance. k = 7 gave a smooth enough boundary. |
| Naive Bayes | Lowest test scores (accuracy 0.947, MCC 0.887) and the most false negatives (3 missed malignant cases), which is the worst kind of mistake for this problem. Likely because of the independence assumption, since radius, perimeter and area are highly correlated. AUC is still 0.987 and it's the most stable model in CV. |
| Support Vector Machine | 0.983 test accuracy with 1 FP and 1 FN, and AUC of 0.9997. Best CV scores on all four metrics (accuracy 0.969, MCC 0.934). The RBF kernel works well on this small, scaled dataset. |
| Overall, winner for your dataset? | SVM. Logistic Regression and KNN scored 1.0 on this test split, but SVM has the best cross-validation results so it should generalise best. Logistic Regression is a close second and easier to interpret. I wouldn't use Naive Bayes here because of the missed malignant cases. |

## Streamlit app

Live app: <https://ml-assignment2-2026aj05629-6ypu2uqcpxg8chw5sggyyh.streamlit.app/>

- Upload a test CSV with the 30 feature columns and `diagnosis` (M/B or 1/0). You can also use or download `test_data.csv` from the sidebar.
- Pick one of the five models from the dropdown.
- Shows accuracy, AUC, precision, recall, F1 and MCC for that model, plus the confusion matrix and classification report.
- "Compare all models" tab shows a table and bar chart for all five models on the uploaded data.

The app only loads the saved models from `model/` and calls `predict` / `predict_proba`. Nothing is retrained.
