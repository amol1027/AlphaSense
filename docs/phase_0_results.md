# Phase 0 — Research Results

## 1. Objective

Evaluate whether market, news sentiment, and Reddit-derived features
provide useful information for predicting the next-hour direction of
RELIANCE and TCS.

---

## 2. Dataset

Assets:
- RELIANCE
- TCS

Feature rows:
- 1,276
- 638 per asset

Trading days:
- 29

Prediction horizon:
- 1 hour

Market resolution:
- 15 minutes

---

## 3. Target

The target is the direction of the next-hour return:

target_direction = 1 if future_return > 0
target_direction = 0 otherwise

The future price is obtained from the same asset/exchange exactly
one hour after the prediction timestamp.

Observations without an exact future price receive no target.

---

## 4. Leakage Controls

The pipeline enforces chronological information availability.

News is eligible only when:

published_at <= prediction_timestamp

Market targets are generated from future prices but future-derived
columns are forbidden from becoming model features.

Model scaling and model parameters are fitted using training data only.

Walk-forward evaluation preserves chronological ordering.

---

## 5. Feature Groups

### Raw Market

- open
- high
- low
- close
- volume

### Engineered Market

- return_15m
- return_30m
- return_1h
- high_low_range
- close_open_return
- volume_change

### News

- sentiment_mean
- sentiment_std
- news_count
- positive_ratio
- negative_ratio

### Reddit

- reddit_sentiment_mean
- reddit_sentiment_std
- reddit_count
- reddit_positive_ratio
- reddit_negative_ratio
- reddit_score_mean
- reddit_comments_mean
- reddit_engagement_mean

---

## 6. Model Evaluation

Models evaluated:

- Logistic Regression
- Random Forest
- HistGradientBoosting

Evaluation used chronological walk-forward testing.

Primary metric:

- balanced accuracy

Additional metrics:

- accuracy
- precision
- recall
- median balanced accuracy
- standard deviation of balanced accuracy

---

## 7. Walk-Forward Findings

The strongest aggregate market-only result was:

Market + Logistic Regression

Mean balanced accuracy:
0.5395

Engineered features did not improve the aggregate result over
the raw-market Logistic Regression benchmark.

However, asset-specific behavior differed substantially.

RELIANCE benefited more from engineered market features.

TCS showed different behavior and did not demonstrate the same
benefit from engineered market features.

News features sometimes received substantial model importance,
but increased feature importance did not consistently translate
into improved out-of-sample performance.

---

## 8. Frozen Candidates

After model and feature exploration, the following candidates
were frozen before the final evaluation.

### RELIANCE

Reduced Engineered Market + Logistic Regression

Features:

- return_15m
- return_30m
- return_1h
- high_low_range
- volume_change

### TCS

Market + News + HistGradientBoosting

Features:

- open
- high
- low
- close
- volume
- sentiment_mean
- sentiment_std
- news_count
- positive_ratio
- negative_ratio

---

## 9. Final Untouched Evaluation

Final test period:

2026-08-03 onward

### RELIANCE

Candidate:
Reduced Engineered Market + Logistic Regression

Samples:
126

Accuracy:
59.52%

Balanced accuracy:
50.96%

Precision:
100.00%

Recall:
1.92%

Confusion matrix:

((74, 0),
 (51, 1))

Majority baseline balanced accuracy:
50.00%

Improvement:
+0.96 percentage points

### TCS

Candidate:
Market + News + HistGradientBoosting

Samples:
126

Accuracy:
52.38%

Balanced accuracy:
50.68%

Precision:
47.22%

Recall:
29.31%

Confusion matrix:

((49, 19),
 (41, 17))

Majority baseline balanced accuracy:
50.00%

Improvement:
+0.68 percentage points

---

## 10. Interpretation

The final untouched evaluation does not demonstrate a robust
directional prediction advantage.

Although both frozen candidates slightly exceeded the 50%
balanced-accuracy baseline, the improvements were small:

- RELIANCE: +0.96 percentage points
- TCS: +0.68 percentage points

The RELIANCE result is particularly affected by highly
imbalanced predictions: the model predicted almost all
observations as the negative class.

Therefore the 59.52% accuracy should not be interpreted as
evidence of a 59.52%-accurate directional forecasting system.

Balanced accuracy provides the more appropriate interpretation.

---

## 11. Research Conclusion

Phase 0 does not establish a robust predictive edge for the
current dataset.

Market-derived features showed more useful and stable behavior
during feature exploration than news or Reddit features.

However, the apparent gains observed during walk-forward
model selection largely disappeared during the untouched
final evaluation.

The current evidence therefore does not justify claiming
that the model can reliably predict next-hour stock direction.

The result should be treated as a baseline research finding,
not as evidence of a production-ready trading signal.

---

## 12. Limitations

- Small dataset: 1,276 feature rows.
- Only two assets.
- Only 29 trading days.
- Final test contains only 126 observations per asset.
- Very short evaluation horizon.
- Model selection and validation are therefore sensitive to
  period-specific behavior.
- Results should not be generalized to other assets, periods,
  or market regimes without additional testing.

---

## 13. Phase 0 Status

Status: COMPLETE

The data pipeline, feature construction, leakage controls,
model evaluation, walk-forward validation, feature ablation,
model comparison, and locked final evaluation have been
implemented and tested.

The next phase should focus on expanding the dataset and
strengthening the experimental design rather than further
optimizing the current small sample. 