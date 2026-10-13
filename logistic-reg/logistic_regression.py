"""
Breast Cancer / Iris — Logistic Regression Classification Pipeline
=======================================================================
Binary classification (breast cancer malignant/benign) with accuracy,
confusion matrix, precision/recall/F1, threshold tuning, and ROC/AUC.
Plus a multiclass extension on Iris.

Run with:
    python logistic_regression.py

Requires: numpy, pandas, matplotlib, scikit-learn (piscine1b_w3 venv)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score,
)


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Exercise 0: Environment and libraries
# ---------------------------------------------------------------------------
def exercise_0() -> None:
    section("EXERCISE 0 — Environment and libraries")
    import sklearn
    import matplotlib as mpl
    print(sklearn.__version__, mpl.__version__)


# ---------------------------------------------------------------------------
# Exercise 1: Load and split
# ---------------------------------------------------------------------------
def exercise_1():
    section("EXERCISE 1 — Load and split")

    data = load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target

    print(X.shape)
    print(f"feature count: {X.shape[1]}")
    print(f"class names:   {data.target_names.tolist()}")
    print(f"class counts:  {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(X_train.shape, X_test.shape)
    print(f"train class counts: {y_train.value_counts().to_dict()}")
    print(f"test class counts:  {y_test.value_counts().to_dict()}")

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    return data, X_train, X_test, y_train, y_test, scaler, X_train_std, X_test_std


# ---------------------------------------------------------------------------
# Exercise 2: Baseline logistic regression
# ---------------------------------------------------------------------------
def exercise_2(X_train_std, X_test_std, y_train, y_test):
    section("EXERCISE 2 — Baseline logistic regression")

    model = LogisticRegression(max_iter=10000, random_state=42)
    model.fit(X_train_std, y_train)

    y_pred_train = model.predict(X_train_std)
    y_pred_test = model.predict(X_test_std)
    print(y_pred_test[:10])

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(f"train accuracy: {train_acc:.4f}")
    print(f"test accuracy:  {test_acc:.4f}")

    proba_test = model.predict_proba(X_test_std)
    print(proba_test.shape)
    print(proba_test[:5].round(4))

    return model, y_pred_test, proba_test


# ---------------------------------------------------------------------------
# Exercise 3: Confusion matrix and metrics
# ---------------------------------------------------------------------------
def exercise_3(y_test, y_pred_test):
    section("EXERCISE 3 — Confusion matrix and metrics")

    cm = confusion_matrix(y_test, y_pred_test)
    print(cm)

    prec_malignant = precision_score(y_test, y_pred_test, pos_label=0)
    rec_malignant = recall_score(y_test, y_pred_test, pos_label=0)
    f1_malignant = f1_score(y_test, y_pred_test, pos_label=0)
    print(f"malignant precision: {prec_malignant:.4f}")
    print(f"malignant recall:    {rec_malignant:.4f}")
    print(f"malignant F1:        {f1_malignant:.4f}")

    print(classification_report(y_test, y_pred_test, target_names=["malignant", "benign"]))

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=["malignant", "benign"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — Breast Cancer (test set)")
    fig.savefig("ex03_confusion.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    return cm


# ---------------------------------------------------------------------------
# Exercise 4: Threshold tuning
# ---------------------------------------------------------------------------
def exercise_4(y_test, proba_test):
    section("EXERCISE 4 — Threshold tuning")

    proba_malignant = proba_test[:, 0]

    for threshold in [0.3, 0.5, 0.7, 0.9]:
        y_pred_t = np.where(proba_malignant >= threshold, 0, 1)
        prec = precision_score(y_test, y_pred_t, pos_label=0)
        rec = recall_score(y_test, y_pred_t, pos_label=0)
        print(f"threshold={threshold:.2f}  precision={prec:.4f}  recall={rec:.4f}")

    proba_benign = proba_test[:, 1]

    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        y_pred_t = (proba_benign >= threshold).astype(int)
        prec = precision_score(y_test, y_pred_t)
        rec = recall_score(y_test, y_pred_t)
        print(f"threshold={threshold:.2f}  precision={prec:.4f}  recall={rec:.4f}")

    return proba_benign


# ---------------------------------------------------------------------------
# Exercise 5: ROC curve and AUC
# ---------------------------------------------------------------------------
def exercise_5(y_test, proba_benign):
    section("EXERCISE 5 — ROC curve and AUC")

    fpr, tpr, thresholds = roc_curve(y_test, proba_benign)
    auc = roc_auc_score(y_test, proba_benign)
    print(f"AUC: {auc:.4f}")
    print(f"number of threshold points: {len(thresholds)}")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="tab:blue", linewidth=2.0, label=f"LogReg (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Chance (AUC = 0.50)")

    ax.set_title("ROC — Breast Cancer (test set)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    fig.savefig("ex05_roc.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Exercise 6: Multiclass classification
# ---------------------------------------------------------------------------
def exercise_6() -> None:
    section("EXERCISE 6 — Multiclass classification")

    iris = load_iris(as_frame=True)
    X_iris = iris.data
    y_iris = iris.target

    print(X_iris.shape, y_iris.shape)
    print(iris.target_names.tolist())
    print(y_iris.value_counts().sort_index().to_dict())

    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
    )

    scaler_i = StandardScaler()
    X_train_i_std = scaler_i.fit_transform(X_train_i)
    X_test_i_std = scaler_i.transform(X_test_i)

    model_i = LogisticRegression(max_iter=10000, random_state=42)
    model_i.fit(X_train_i_std, y_train_i)

    y_pred_i = model_i.predict(X_test_i_std)
    acc_i = accuracy_score(y_test_i, y_pred_i)
    cm_i = confusion_matrix(y_test_i, y_pred_i)
    print(f"test accuracy: {acc_i:.4f}")
    print(cm_i)

    mismatch_mask = (y_test_i.values == 2) & (y_pred_i == 1)
    mismatch_idx = np.where(mismatch_mask)[0]
    print(f"misclassified test indices: {mismatch_idx}")

    for i in mismatch_idx:
        print(f"sample {i}: true={y_test_i.values[i]}, "
              f"predicted={y_pred_i[i]}, "
              f"probas={model_i.predict_proba(X_test_i_std[[i]]).round(4)}")

    print(classification_report(y_test_i, y_pred_i, target_names=iris.target_names))


# ---------------------------------------------------------------------------
def main() -> None:
    exercise_0()
    data, X_train, X_test, y_train, y_test, scaler, X_train_std, X_test_std = exercise_1()
    model, y_pred_test, proba_test = exercise_2(X_train_std, X_test_std, y_train, y_test)
    exercise_3(y_test, y_pred_test)
    proba_benign = exercise_4(y_test, proba_test)
    exercise_5(y_test, proba_benign)
    exercise_6()


if __name__ == "__main__":
    main()
