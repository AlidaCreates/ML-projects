# Breast Cancer / Iris — Logistic Regression Classification

Binary classification on the Breast Cancer Wisconsin dataset (malignant
vs. benign), covering accuracy, confusion matrix, precision/recall/F1,
clinically-motivated threshold tuning, ROC/AUC — plus a multiclass
extension on Iris (3 species).

## Verified end-to-end

Both datasets are built into scikit-learn (`load_breast_cancer`,
`load_iris`) — no network access needed — so `logistic_regression.py`
was run fully. **Every printed number matches the exercise sheet's
expected output exactly**: train/test accuracy (0.9890/0.9825),
predicted probabilities, the confusion matrix `[[41,1],[1,71]]`,
malignant precision/recall/F1 (all 0.9762), the full threshold-tuning
tables (both framings), AUC (0.9954), and the multiclass results
including the exact misclassified sample index (23) and its predicted
probabilities.

## How to run

```bash
# activate the persistent piscine1b_w3 venv
pip install matplotlib
pip freeze > requirements.txt
python logistic_regression.py
```

## Pipeline

1. **Ex 0** — verify scikit-learn / matplotlib install.
2. **Ex 1** — load breast cancer data (30 features, 569 samples), stratified 80/20 split, standardize.
3. **Ex 2** — fit `LogisticRegression(max_iter=10000)`, get class predictions and `predict_proba`.
4. **Ex 3** — confusion matrix, malignant-class precision/recall/F1 (`pos_label=0`), full classification report, confusion matrix plot.
5. **Ex 4** — threshold tuning in both framings (malignant-as-positive and the standard benign-as-positive convention), showing the precision/recall trade-off.
6. **Ex 5** — ROC curve and AUC on the benign-positive convention, ROC plot with chance line.
7. **Ex 6** — same pipeline on Iris (3-class), confusion matrix, inspect the one ambiguous versicolor/virginica misclassification, classification report.

## Key results

| Metric | Train | Test |
|---|---|---|
| Accuracy | 0.9890 | 0.9825 |

**Confusion matrix (test):** 41 TN, 1 FP, 1 FN, 71 TP — malignant
precision/recall/F1 all 0.9762 (1 of 42 malignant cases missed).

**AUC = 0.9954** — the model separates the two classes almost perfectly.

**Threshold tuning, clinical framing:** the clinically relevant metric is
malignant-class recall — you don't want to miss a cancer case. At
threshold 0.30 (predicting malignant more liberally), recall is 0.9762
(1 missed case) at a precision cost (0.8913). At threshold 0.90, recall
falls to 0.8810 (5 missed cases) even though precision hits 1.0. A
screening-oriented clinical team would pick a **lower** threshold to
maximize recall, accepting more false alarms in exchange for catching
more actual cancers.

**Multiclass (Iris):** 93.33% test accuracy. Setosa is perfectly
separated; the only error is one true-virginica sample predicted as
versicolor (sample index 23), where the model itself is genuinely
uncertain — 51.45% versicolor vs. 48.42% virginica — the canonical fuzzy
boundary between those two species.

## Files

| File | Description |
|---|---|
| `logistic_regression.py` | Complete script, one function per exercise (0–6) |
| `requirements.txt` | Frozen package versions |
| `ex03_confusion.png` | Confusion matrix heatmap, breast cancer test set |
| `ex05_roc.png` | ROC curve with AUC and chance-line reference |

## Notes for audit

- `pos_label=0` is used deliberately for the malignant-class metrics in
  Exercise 3/4 since sklearn's default positive class is 1 (benign) —
  worth being able to explain why that flag is there if asked.
- The two threshold-tuning loops in Exercise 4 use **opposite
  directions** on purpose: the first thresholds `proba[:, 0]` (malignant
  probability) so precision rises and recall falls as the threshold
  rises for malignant; the second thresholds `proba[:, 1]` (benign
  probability) using the standard sklearn convention, where the
  precision/recall trade-off runs the familiar direction for the
  positive class. Both are correct — they're just two different framings
  of the same decision boundary.
