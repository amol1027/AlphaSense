# Phase 1 Results

## Status

**Phase 1 is frozen.**

Phase 1 expanded the research dataset and re-ran the frozen Phase 0 methodology without changing the selected models or evaluation procedure.

The purpose of this phase was to determine whether the Phase 0 findings would persist on a substantially larger dataset.

---

## 1. Dataset

The final Phase 1 feature dataset contains:

- **28,564 feature rows**
- **14,282 RELIANCE rows**
- **14,282 TCS rows**
- Market data spanning **2024-01-01 through 2026-08-21**
- 15-minute market observations
- Next-hour binary direction target
- Historical news sentiment features

The final locked evaluation period begins:

```text
2026-08-10 00:00:00 UTC
````

The final holdout contains:

* **10 trading sessions**
* **210 usable test observations per asset**
* **440 raw test rows**
* **20 missing targets**

The missing targets occur at observations where the future one-hour observation required to construct the target is unavailable.

Candidate feature coverage on the final test period was complete.

---

## 2. Frozen Candidates

No model tuning was performed during the final locked evaluation.

### RELIANCE

```text
Reduced Engineered Market
+
Logistic Regression
```

Features:

```text
return_15m
return_30m
return_1h
high_low_range
volume_change
```

### TCS

```text
Market + News
+
HistGradientBoosting
```

Features:

```text
open
high
low
close
volume
sentiment_mean
sentiment_std
news_count
positive_ratio
negative_ratio
```

### Baseline

A majority-class classifier was used as the benchmark.

---

## 3. Locked Evaluation

### RELIANCE

Candidate:

```text
Accuracy:            45.71%
Balanced accuracy:   47.95%
```

Majority baseline:

```text
Accuracy:            47.62%
Balanced accuracy:   50.00%
```

Difference:

```text
Balanced accuracy delta: -2.05 percentage points
```

Confusion matrix:

```text
TN: 95
FP:  5
FN: 109
TP:  1
```

The RELIANCE model predicted almost all observations as DOWN.

---

### TCS

Candidate:

```text
Accuracy:            45.71%
Balanced accuracy:   44.66%
```

Majority baseline:

```text
Accuracy:            51.43%
Balanced accuracy:   50.00%
```

Difference:

```text
Balanced accuracy delta: -5.34 percentage points
```

Confusion matrix:

```text
TN: 88
FP: 20
FN: 94
TP:  8
```

---

## 4. Daily Evaluation

The locked test period was evaluated separately by trading day.

### RELIANCE

```text
Days: 10
Positive days: 0
Mean candidate balanced accuracy: 47.63%
Mean baseline balanced accuracy:  50.00%
Mean delta:                       -2.37 pp
```

The candidate did not outperform the baseline on any of the ten test days.

The weakest days were:

```text
2026-08-17: -8.65 pp
2026-08-18: -10.00 pp
```

### TCS

```text
Days: 10
Positive days: 1
Mean candidate balanced accuracy: 42.99%
Mean baseline balanced accuracy:  50.00%
Mean delta:                       -7.01 pp
```

The candidate outperformed the baseline on only one of the ten test days.

The largest negative daily difference occurred on:

```text
2026-08-13: -41.67 pp
```

---

## 5. Prediction Distribution

The prediction distribution diagnostic showed substantial prediction collapse.

### RELIANCE

```text
Test observations:       210
Predicted DOWN:          204
Predicted UP:              6
Predicted UP rate:      2.86%
```

Actual UP rate:

```text
52.38%
```

### TCS

```text
Test observations:       210
Predicted DOWN:          182
Predicted UP:             28
Predicted UP rate:      13.33%
```

Actual UP rate:

```text
48.57%
```

This demonstrates that the weak balanced accuracy is not simply caused by a small difference in overall class frequencies.

---

## 6. Probability Diagnostic

The probability diagnostic showed very weak separation between the two target classes.

### RELIANCE

Test P(UP):

```text
Minimum:  0.424812
Median:   0.465073
Maximum:  0.531794
Mean:     0.466071
```

Mean predicted probability by actual class:

```text
Actual DOWN: 0.468705
Actual UP:   0.463676
```

The model therefore assigned slightly lower probabilities to actual UP observations than to DOWN observations.

### TCS

Test P(UP):

```text
Minimum:  0.315695
Median:   0.460741
Maximum:  0.555438
Mean:     0.445182
```

Mean predicted probability by actual class:

```text
Actual DOWN: 0.445628
Actual UP:   0.444709
```

The two classes have essentially identical predicted probabilities.

### Interpretation

The issue is therefore not simply the default 0.50 classification threshold.

Changing the threshold would change the number of positive predictions, but the probability distributions do not show meaningful ranking or separation between future UP and DOWN observations.

Threshold tuning was therefore not performed.

---

## 7. Feature-vs-Target Diagnostic

The feature-vs-target diagnostic was performed using the training period only.

The locked final test observations were excluded.

### RELIANCE

The strongest individual market features were:

```text
return_1h       AUC: 0.5141
high_low_range  AUC: 0.5254
return_15m      AUC: 0.5052
return_30m      AUC: 0.5049
```

News features were effectively at chance:

```text
sentiment_mean   AUC: 0.5010
news_count       AUC: 0.5005
positive_ratio   AUC: 0.5013
negative_ratio   AUC: 0.4993
```

### TCS

The individual features were also effectively at chance.

Examples:

```text
return_1h       AUC: 0.4921
return_30m      AUC: 0.4924
high_low_range  AUC: 0.5030
sentiment_mean  AUC: 0.5003
news_count      AUC: 0.5008
```

These results indicate very weak marginal relationships between the current feature set and next-hour direction.

---

## 8. Data Integrity and Leakage Audits

Several independent checks were completed before freezing the Phase 1 conclusion.

### Feature dataset audit

```text
Rows:                       28,564
Invalid timestamps:              0
Null timestamps:                 0
Duplicate feature keys:          0
Invalid assets:                  0
Invalid exchanges:               0
Invalid numeric values:          0
Infinite numeric values:         0
Invalid targets:                 0
Invalid sentiment features:      0
Invalid Reddit features:         0
Weekend predictions:             0
Invalid target alignment:        0

STATUS: PASS
```

### Market feature cutoff audit

All independently reconstructed market features matched:

```text
return_15m
return_30m
return_1h
high_low_range
close_open_return
volume_change
```

Maximum difference:

```text
0.000000000000
```

Audit result:

```text
PASS
```

### News feature cutoff audit

The news features were independently reconstructed using the one-hour information window.

All five features matched:

```text
sentiment_mean
sentiment_std
news_count
positive_ratio
negative_ratio
```

Maximum difference:

```text
0.000000000000
```

Future-news cutoff violations:

```text
0
```

Audit result:

```text
PASS
```

### Target construction

The target is defined using the close exactly one hour after the prediction timestamp:

```text
target_return =
    (future_close - close) / close
```

and:

```text
target_direction =
    1 if target_return > 0
    0 otherwise
```

Target alignment was independently inspected and no target-construction defect was identified.

---

## 9. Reddit Data Limitation

Phase 1 does not contain real historical Reddit data.

The current Reddit source is a sample/fixture dataset and therefore Reddit results must not be interpreted as evidence about the predictive value of real Reddit sentiment.

Consequently:

> Phase 1 conclusions about market and news features are valid for the available datasets, but no conclusion about real-world Reddit predictive value should be made.

---

## 10. Final Phase 1 Conclusion

The expanded Phase 1 dataset did not recover the predictive performance observed during earlier model-development experiments.

The frozen candidates performed at or below the majority baseline on the untouched final holdout:

```text
RELIANCE:
47.95% balanced accuracy
vs
50.00% baseline

TCS:
44.66% balanced accuracy
vs
50.00% baseline
```

The probability diagnostic additionally showed negligible separation between future UP and DOWN observations.

The feature-vs-target diagnostic showed that most individual features have AUC values close to 0.50, indicating weak marginal predictive relationships.

The major data-integrity checks passed:

```text
Target alignment              PASS
Market feature cutoff         PASS
News information cutoff      PASS
Feature data quality          PASS
Final test feature coverage   PASS
```

Therefore, the current evidence does **not** support a claim that the frozen Phase 1 feature set provides reliable next-hour directional prediction for RELIANCE or TCS.

The most defensible conclusion is:

> **On the expanded Phase 1 dataset, the selected market and news features exhibit little stable predictive information for next-hour UP/DOWN classification. The frozen models fail to demonstrate out-of-sample improvement over the majority baseline on the untouched final holdout, and their predicted probabilities show negligible class separation.**

This is treated as a **negative research result**, not as evidence of an implementation failure.

---

## 11. Phase 1 Freeze Rules

Phase 1 is now frozen.

The following will not be changed to improve the locked result:

* final test period
* target definition
* frozen candidate definitions
* model hyperparameters
* prediction threshold
* final-test feature selection
* final-test data processing

Any future experimentation must use a new research phase and must not alter the locked Phase 1 evaluation.

---

## 12. Recommended Phase 2 Direction

Phase 2 should not begin with additional hyperparameter tuning against the Phase 1 holdout.

Instead, future research should investigate the underlying research design, including:

1. Alternative prediction horizons.
2. Alternative target definitions.
3. Richer temporal market features.
4. Regime/context features.
5. More appropriate news representations.
6. Larger and more diverse asset coverage.
7. Real historical Reddit data, if obtainable.
8. Additional walk-forward validation on future data.

The objective should be to determine whether the weak Phase 1 result is caused by:

```text
target definition
        OR
feature representation
        OR
market regime
        OR
insufficient information
        OR
lack of predictable short-horizon directional signal
```

rather than simply optimizing the existing models against the same data.

---

## Phase 1 Status

**FROZEN**

```text
Data expansion                 PASS
Feature construction           PASS
Temporal cutoff audits         PASS
Target alignment               PASS
Final locked evaluation        COMPLETE
Daily evaluation               COMPLETE
Probability diagnostic         COMPLETE
Feature signal diagnostic      COMPLETE
Research conclusion            COMPLETE
```
