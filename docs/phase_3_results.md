# Phase 3 — Conditional Signal Validation (IN PROGRESS)

## Frozen input (from Phase 2)

- Horizon 1h, 3-class DOWN/NEUTRAL/UP, threshold ±0.203666%
- 60m news window, core feature `sentiment_mean`
- Regime: high sentiment magnitude = `abs(sentiment_mean)` >= train P75
- Boundaries: TRAIN 2026-07-05, OOS 2026-07-25..2026-08-10, LOCKED 2026-08-10+ excluded
- Locked rows never used for thresholds, fitting, or comparison

## 3.1 Daily walk-forward (done)

`scripts/validate_phase3_magnitude_stability.py` — expanding train, test = single day, guardrails >=30 train / >=10 test event rows.

| Asset | Folds | Median BA delta | Verdict |
|---|---|---:|---|
| RELIANCE | 2/10 | +0.0000 | FAIL (sparse, no lift) |
| TCS | 0/10 | — | FAIL (no evaluable folds) |

## 3.2 Expanding pooled + bootstrap (done)

`scripts/validate_phase3_magnitude_pooled.py` — train `[TRAIN_START, D)`, test `[D, LOCKED)` pooled, B=1000 paired bootstrap on BA delta.
Pre-reg PASS = >=8 folds AND median BA delta > +2% AND win rate >= 60%.

| Asset | Folds | Median BA delta | Win rate | Verdict |
|---|---|---:|---:|---|
| RELIANCE | 6/10 | +2.50% | 83% (5/6) | FAIL (fold count; CIs span 0) |
| TCS | 5/10 | +14.17% | 100% (5/5) | FAIL (fold count; N=20–27, wide CIs) |

Pooled Phase 2.7 lift (+4.9pp RELIANCE / +14.2pp TCS on 61/27 rows) does not meet the stability bar. Directionally consistent but underpowered — do not promote to general feature.

## 3.3 Market-regime interaction (done)

`scripts/validate_phase3_regime_interaction.py` — same frozen config; VOL (`high_low_range` >= train median) and TREND (`abs(return_1h)` >= train median) buckets, train-on-all-event / test-per-bucket, guardrails >=30 train / >=15 per bucket. Pre-reg PASS per bucket = >=8 folds AND median > +2% AND win >= 60%.

| Asset | Folds | vol_hi | vol_lo | mov_hi | mov_lo | Verdict |
|---|---|---|---|---|---|---|
| RELIANCE | 2/10 | 1 fold | 2 folds | 1 fold | 2 folds | FAIL (coverage) |
| TCS | 0/10 | — | — | — | — | FAIL (no evaluable folds) |

Single-fold hints (RELIANCE 2026-07-27: vol_hi +25pp on n=17, mov_hi +13.9pp) are anecdotal at this N. Interaction does not rescue the regime.

## Phase 3 conclusion

Magnitude-regime is **exploratory only — do not promote**. All three cuts fail stability/coverage. Next step is power expansion (more assets/history, re-run frozen 3.2 unchanged), not new features or modalities.

## 3.4 Power expansion (done 2026-09-25)

Added HDFCBANK (`NSE_EQ|INE040A01034`), INFY (`NSE_EQ|INE009A01021`), ICICIBANK (`NSE_EQ|INE090A01021`).

- Market: all 3 downloaded 2024-01-01→2026-09-23 (16,87x rows each). 1 provider OHLC violation quarantined (HDFCBANK 2024-06-25 09:15 IST, low 837.4 > open 835.6); validation itself stays strict. Combined market now 83,259 rows, 0 duplicates.
- News: Marketaux complete for all 5 assets; GDELT throttled (HTTP 429) after 235 windows / 395 failed (timeouts). New-asset yield negligible: HDFCBANK 13, ICICIBANK 6, INFY 5 articles (vs RELIANCE 1218, TCS 258). Expanded `research_news.csv` (backup in temp dir) promoted; FinBERT re-run over 1,500 articles; `phase1_features.csv` rebuilt (69,530 valid targets).
- Frozen 3.2 re-run on 5 assets: HDFCBANK/ICICIBANK/INFY 0/10 folds (news too sparse); RELIANCE 3/10, median +9.7pp, 100% win → FAIL; TCS 0/10 → FAIL. Note old-asset fold counts shifted (RELIANCE 6→3, TCS 5→0) after a 1-article TCS dedup — further evidence of fragility.

News-side power expansion is blocked on provider coverage, not effort. Market-only 5-asset baselines remain available work.

## 3.5 Market-only 5-asset baselines (done)

`scripts/validate_phase3_market_baselines.py` — binary next-hour direction, normalized features, majority vs logistic, expanding monthly walk-forward from 2024-07 (locked 2026-08-10+ excluded), guardrails >=500 train / >=50 test rows. Pre-reg PASS = >=8 folds AND median BA delta > +2% AND win >= 60%.

| Asset | Folds | Median BA delta | Win rate | Locked (base vs LR) | Verdict |
|---|---|---:|---:|---|---|
| HDFCBANK | 26 | +0.04pp | 54% | 0.5000 vs 0.4910 | FAIL |
| ICICIBANK | 26 | +0.03pp | 50% | 0.5000 vs 0.4822 | FAIL |
| INFY | 26 | +0.36pp | 62% | 0.5000 vs 0.4835 | FAIL |
| RELIANCE | 26 | +0.03pp | 50% | 0.5000 vs 0.4795 | FAIL |
| TCS | 26 | +0.19pp | 62% | 0.5000 vs 0.4910 | FAIL |
| POOLED | 130 | +0.1pp | 55% | — | FAIL |

Logistic underperforms majority on the locked holdout for all 5 assets. With 130 folds the median delta is a tenth of a point — the next-hour direction question has no demonstrable edge in this feature family at this scale. Recommend freezing next-hour direction research and pivoting (different horizon/target or different problem) over further feature iteration.
