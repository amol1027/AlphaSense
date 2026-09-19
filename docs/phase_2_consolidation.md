# Phase 2 — Signal Research Consolidation

## 1. Phase Objective

Phase 2 investigated whether alternative target definitions, prediction horizons, financial-news sentiment, and selective news regimes provide measurable predictive information for next-hour stock-direction prediction.

The research focused on RELIANCE and TCS and preserved the locked evaluation boundary established in Phase 0 and Phase 1.

The primary purpose of Phase 2 was not to maximize model performance through repeated tuning, but to determine whether additional signal sources demonstrated sufficient evidence to justify carrying them into subsequent research.

---

## 2. Experiments Completed

### 2.1 Prediction Horizon Research

Prediction horizons of:

- 30 minutes
- 1 hour
- 2 hours
- 4 hours

were investigated.

No horizon demonstrated a strong and consistently reliable predictive advantage.

The 4-hour horizon showed somewhat larger deviations in some univariate analyses, but the evidence remained weak.

The 1-hour horizon therefore remained the primary research horizon because it was already established as the project target and no alternative horizon demonstrated sufficient evidence to replace it.

---

### 2.2 Three-Class Target Research

A three-class target was introduced:

- DOWN
- NEUTRAL
- UP

using a frozen pooled P50 absolute 1-hour return threshold of:

**±0.203666%**

The resulting class distributions were approximately:

| Asset | DOWN | NEUTRAL | UP |
|---|---:|---:|---:|
| RELIANCE | 26.14% | 49.94% | 23.92% |
| TCS | 26.75% | 50.06% | 23.19% |

The P75 threshold was rejected as the primary target because it produced approximately 75% neutral observations.

The three-class formulation was retained as a useful research target, although overall predictive performance remained weak.

---

## 3. Market-Only Signal

The Phase 2 experiments confirmed the weakness already observed in Phase 1.

Market features did not demonstrate a strong, consistent predictive relationship with the next-hour three-class target.

Probability diagnostics showed only weak class-ranking behaviour.

For example:

### RELIANCE

- DOWN AUC: 0.545
- NEUTRAL AUC: 0.582
- UP AUC: 0.598

### TCS

- DOWN AUC: 0.496
- NEUTRAL AUC: 0.539
- UP AUC: 0.558

These results indicate that the market feature set contains some weak ranking information in portions of the data, but not enough evidence to characterize it as a robust predictive signal.

---

## 4. News Contribution

Phase 2.3 compared:

- Market-only
- Market + News

using matched observations so that differences in sample availability did not confound the comparison.

The news features were based on a 60-minute aggregation window.

### RELIANCE

| Metric | Market-only | Market + News | Change |
|---|---:|---:|---:|
| Accuracy | 0.6593 | 0.6741 | +0.0148 |
| Balanced Accuracy | 0.3260 | 0.3812 | +0.0552 |
| Macro-F1 | 0.2649 | 0.3536 | +0.0887 |

### TCS

| Metric | Market-only | Market + News | Change |
|---|---:|---:|---:|
| Accuracy | 0.3056 | 0.2778 | -0.0278 |
| Balanced Accuracy | 0.2771 | 0.2662 | -0.0109 |
| Macro-F1 | 0.2222 | 0.2339 | +0.0117 |

The results therefore did not support the conclusion that adding news produces a universal improvement.

The effect differed between assets.

---

## 5. News Window Research

News aggregation windows of:

- 30 minutes
- 60 minutes
- 120 minutes
- 240 minutes

were evaluated.

For RELIANCE, the 60-minute window produced the strongest results across Accuracy, Balanced Accuracy, and Macro-F1.

The 60-minute window was therefore frozen for subsequent Phase 2 research.

No evidence justified replacing it with a longer or shorter window.

---

## 6. News Intensity Research

Additional news features were investigated:

- news_count
- news_burst
- sentiment_std
- positive_ratio
- negative_ratio
- sentiment_imbalance
- source_diversity

These features were evaluated as an additive news-information block.

The results were not consistent across assets.

### RELIANCE

- Accuracy: +0.0492
- Balanced Accuracy: +0.0065
- Macro-F1: +0.0409

### TCS

- Accuracy: -0.0976
- Balanced Accuracy: -0.0814
- Macro-F1: -0.0753

The evidence therefore did not justify treating these intensity and composition features as universally useful predictive features.

The possibility that news density interacts with market regimes remains a research hypothesis, but the Phase 2 experiments did not establish it as a general predictive mechanism.

---

## 7. Selective News Regime Research

Phase 2.7 investigated whether news becomes more informative under specific event conditions.

Three event definitions were evaluated:

1. High news volume
2. High sentiment magnitude
3. High sentiment dispersion

All event thresholds were derived from the development period only and then frozen before evaluation on the OOS period.

The locked Aug 10+ evaluation period was not used.

---

## 8. High News Volume

High news volume did not demonstrate a positive incremental effect for RELIANCE.

### RELIANCE

- Market Balanced Accuracy: 0.3137
- Market + News Balanced Accuracy: 0.2157
- Change: -0.0980

Macro-F1 also decreased:

- Market-only: 0.2883
- Market + News: 0.2529
- Change: -0.0354

The TCS high-volume regime contained only 5 OOS observations and was therefore not treated as sufficient evidence for evaluation.

High news volume was not carried forward as a validated conditional signal.

---

## 9. High Sentiment Magnitude

High sentiment magnitude produced the clearest conditional evidence in Phase 2.7.

The event condition was defined using the development-period P75 of:

**abs(sentiment_mean)**

### RELIANCE

- Market Balanced Accuracy: 0.3129
- Market + News Balanced Accuracy: 0.3622
- Change: +0.0493

- Market Macro-F1: 0.2866
- Market + News Macro-F1: 0.3237
- Change: +0.0371

The corresponding OOS event sample contained 61 observations.

### TCS

- Market Balanced Accuracy: 0.1417
- Market + News Balanced Accuracy: 0.2833
- Change: +0.1417

- Market Macro-F1: 0.1462
- Market + News Macro-F1: 0.2602
- Change: +0.1141

The corresponding OOS event sample contained 27 observations.

For TCS, DOWN recall increased from 0.30 to 0.60 and UP recall increased from 0.125 to 0.25.

However, NEUTRAL recall remained 0.

The TCS result is therefore promising as a research observation but is based on a small OOS event sample and should not be treated as established generalization.

---

## 10. High Sentiment Dispersion

High sentiment dispersion did not demonstrate a positive incremental effect for RELIANCE.

### RELIANCE

- Market Balanced Accuracy: 0.2821
- Market + News Balanced Accuracy: 0.2051
- Change: -0.0769

- Market Macro-F1: 0.2444
- Market + News Macro-F1: 0.1905
- Change: -0.0540

The TCS high-dispersion regime contained only 7 OOS observations and was therefore skipped as insufficient for meaningful evaluation.

High sentiment dispersion was not carried forward as a validated conditional signal.

---

## 11. Phase 2 Evidence Summary

| Research Component | Phase 2 Finding | Carry Forward |
|---|---|---|
| 1-hour horizon | Established primary horizon; alternatives did not show sufficient advantage | Yes |
| Three-class target | Useful research formulation | Yes |
| ±0.203666% threshold | Frozen P50 threshold | Yes |
| 60-minute news window | Strongest RELIANCE window | Yes |
| sentiment_mean | Core news sentiment feature | Yes |
| News count | Inconsistent | No |
| News burst | Inconsistent | No |
| Sentiment dispersion | Inconsistent | No |
| Positive/negative ratios | No universal improvement established | No |
| Sentiment imbalance | No universal improvement established | No |
| Source diversity | No universal improvement established | No |
| High news volume regime | No positive consistent evidence | No |
| High sentiment magnitude | Conditional positive evidence | Research signal |
| High sentiment dispersion | No positive consistent evidence | No |

---

## 12. What Phase 2 Supports

Phase 2 provides evidence for the following conclusions:

1. The existing market feature set does not provide strong universal predictive performance for the selected next-hour three-class target.

2. Financial-news sentiment can provide incremental information in some settings.

3. The incremental contribution of news is asset-dependent rather than universal.

4. A 60-minute news aggregation window is the frozen window for subsequent research.

5. High sentiment magnitude is the strongest conditional news regime identified during Phase 2.

6. The high-sentiment-magnitude result should be treated as a conditional research signal rather than a universally predictive feature.

---

## 13. What Phase 2 Does Not Support

Phase 2 does not support the following claims:

- that news universally improves prediction;
- that higher news volume is inherently predictive;
- that sentiment dispersion is universally predictive;
- that the additional news-intensity features should always be included;
- that high sentiment magnitude has been proven to generalize;
- that the Phase 2 results establish deployable trading performance.

The observed improvements are experimental findings within the defined development and OOS periods.

They require further validation before being treated as robust predictive relationships.

---

## 14. Locked Evaluation Protection

The locked evaluation boundary begins on:

**2026-08-10**

The Phase 2 experiments and event-regime audits did not use the locked observations for threshold selection or model comparison.

The Phase 2.7 audit explicitly verified:

- OOS timestamps matched between Market-only and Market + News;
- locked observations were not included;
- event thresholds were derived from the training/development period;
- event observations were evaluated using matched samples.

The locked evaluation therefore remains available for a future final evaluation.

---

## 15. Phase 2 Frozen Research Configuration

The following configuration is carried forward:

### Prediction

- Primary horizon: **1 hour**
- Target: **3-class DOWN / NEUTRAL / UP**
- Threshold: **±0.203666%**

### News

- Aggregation window: **60 minutes**
- Core sentiment feature: **sentiment_mean**

### Conditional Research Signal

- High sentiment magnitude:
  **abs(sentiment_mean)** above the frozen development-period P75 threshold.

The high-sentiment-magnitude condition is retained as a research hypothesis/conditional signal rather than as a universally active predictive feature.

---

## 16. Phase 2 Final Conclusion

Phase 2 did not establish a universally predictive contribution from financial news or social-media sentiment.

The experiments instead provide evidence for a more conditional interpretation.

Market features alone remain weak. News sentiment produced measurable incremental changes for RELIANCE under matched-sample evaluation, while the overall effect for TCS was not consistently positive. Additional news-intensity features did not provide universal improvement.

The strongest conditional result was observed when sentiment magnitude was unusually high. In this regime, Market + News outperformed the matched Market-only baseline on Balanced Accuracy and Macro-F1 for both RELIANCE and TCS during the Phase 2 experiment.

However, the TCS result was based on only 27 OOS event observations, and the event-class distributions changed between development and OOS periods. Therefore, the result should be treated as a research-supported conditional signal rather than proof of general predictive power.

Phase 2 therefore establishes the following research position:

> **News sentiment should not be treated as a universally additive predictor. Its potential contribution appears conditional on the information regime, with high sentiment magnitude providing the strongest evidence for further investigation.**

---

## 17. Handoff to Phase 3

Phase 3 should build on the frozen evidence rather than reopen the Phase 2 search space.

The next phase should therefore focus on validating whether the conditional news signal survives a broader and more disciplined evaluation framework.

The following should remain frozen unless new evidence explicitly justifies reopening them:

- 1-hour horizon
- three-class target
- ±0.203666% threshold
- 60-minute news window
- development/OOS/locked temporal boundaries
- no use of locked observations for development decisions

Phase 3 should investigate the stability, generalization, and practical contribution of the identified conditional signal rather than continue searching for arbitrary thresholds or feature combinations.