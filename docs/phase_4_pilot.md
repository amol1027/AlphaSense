# Phase 4 Pilot — Next-Hour Range-Regime Prediction (POSITIVE)

New question (direction frozen as negative): predict whether the next hour's
trading range will be above or below median — a volatility-regime proxy.

## Frozen config

- Target: `fut_range = (max(high) − min(low)) / close` over next 4 bars (1h);
  label = above per-fold train median; incomplete future windows excluded.
- Features (all ≤ t): normalized market set + trailing range mean/max and
  return std over past 4/16 bars, all lagged (no look-ahead).
- Majority vs LogisticRegression; expanding monthly walk-forward from 2024-07;
  locked 2026-08-10+ scored once. Pre-reg PASS = ≥8 folds, median BA delta
  > +2%, win rate ≥ 60%.

## Result: PASS on all 5 assets (130/130 folds)

| Asset | Folds | Median BA delta | Win | Mean LR BA | Locked base vs LR |
|---|---|---:|---:|---:|---|
| HDFCBANK | 26 | +13.5pp | 100% | 0.633 | 0.5000 vs 0.6821 |
| ICICIBANK | 26 | +10.7pp | 100% | 0.608 | 0.5000 vs 0.5411 |
| INFY | 26 | +14.1pp | 100% | 0.637 | 0.5000 vs 0.6732 |
| RELIANCE | 26 | +14.3pp | 100% | 0.636 | 0.5000 vs 0.5390 |
| TCS | 26 | +14.8pp | 100% | 0.650 | 0.5000 vs 0.6602 |
| POOLED | 130 | +13.9pp | 100% | — | — |

Volatility clusters; direction doesn't. This is the project's first signal to
survive walk-forward + locked evaluation. Next: ablate trailing features,
test horizon sensitivity (30m/2h/4h), and check whether news adds anything on
top of the volatility baseline — with the same pre-reg discipline.

## 4.2 Ablation (done) — median walk-forward BA | locked BA

`scripts/validate_phase4_ablation.py`, 26 folds/asset throughout.

| Group | HDFCBANK | ICICIBANK | INFY | RELIANCE | TCS |
|---|---|---|---|---|---|
| full (12) | 0.635 \| 0.682 | 0.607 \| 0.541 | 0.641 \| 0.673 | 0.643 \| 0.539 | 0.648 \| 0.660 |
| market_only (6) | 0.627 \| 0.677 | 0.592 \| 0.540 | 0.624 \| 0.621 | 0.619 \| 0.564 | 0.631 \| 0.591 |
| trailing_only (6) | 0.603 \| 0.637 | 0.585 \| 0.543 | 0.602 \| 0.639 | 0.596 \| 0.495 | 0.622 \| 0.605 |
| no_4h (9) | 0.652 \| 0.670 | 0.609 \| 0.548 | 0.641 \| 0.681 | 0.635 \| 0.544 | 0.662 \| 0.669 |
| no_1h (9) | 0.621 \| 0.687 | 0.593 \| 0.530 | 0.614 \| 0.625 | 0.627 \| 0.537 | 0.621 \| 0.593 |
| no_range (8) | 0.636 \| 0.688 | 0.603 \| 0.533 | 0.631 \| 0.649 | 0.628 \| 0.568 | 0.639 \| 0.634 |
| no_retstd (10) | 0.629 \| 0.683 | 0.611 \| 0.539 | 0.632 \| 0.667 | 0.642 \| 0.519 | 0.648 \| 0.660 |

Reading: current-bar range + 1h trailing carry the signal (`no_4h` matches or
beats full on 4/5 assets); the 4h window is redundant; trailing-only underperforms
market-only everywhere. Frozen minimal set: **market + 1h trailing (9 feats)**.

## 4.3 Horizon sensitivity (done) — median BA delta | win | locked delta

`scripts/validate_phase4_horizons.py`, frozen 9-feat set.

| Horizon | HDFCBANK | ICICIBANK | INFY | RELIANCE | TCS |
|---|---|---|---|---|---|
| 30m | +0.162 \| 100% \| +0.168 | +0.133 \| 100% \| +0.053 | +0.141 \| 100% \| +0.143 | +0.129 \| 100% \| +0.038 | +0.162 \| 100% \| +0.137 |
| 1h | +0.152 \| 100% \| +0.170 | +0.109 \| 96% \| +0.048 | +0.141 \| 100% \| +0.181 | +0.135 \| 100% \| +0.044 | +0.162 \| 100% \| +0.169 |
| 2h | +0.116 \| 100% \| +0.165 | +0.093 \| 96% \| +0.021 | +0.115 \| 100% \| +0.146 | +0.127 \| 92% \| −0.020 | +0.136 \| 100% \| +0.203 |
| 4h | +0.071 \| 88% \| +0.167 | +0.073 \| 84% \| −0.031 | +0.098 \| 96% \| +0.084 | +0.078 \| 92% \| −0.128 | +0.066 \| 96% \| +0.076 |

Edge decays monotonically with horizon; at 4h locked flips negative for
RELIANCE/ICICIBANK. 30m ≈ 1h (no gain for extra noise). **1h stays frozen.**

## 4.4 News on top of volatility baseline (done) — the fair rematch

`scripts/validate_phase4_news_on_top.py`: frozen 9-feat base vs base+news
(60m window, matched news-supported rows), RELIANCE + TCS only, expanding
cutoffs over OOS, guardrails ≥30 train / ≥20 test. Pre-reg PASS = ≥8 folds,
median BA delta > +2%, win ≥ 60%.

| Asset | Folds | Median BA delta | Win | Verdict |
|---|---|---:|---:|---|
| RELIANCE | 10/10 | +0.55pp | 50% | FAIL |
| TCS | 8/10 | −3.03pp | 25% | FAIL |

RELIANCE shows +3 to +7pp in early (large-N) folds but collapses to −19pp in
late tiny-N folds; TCS is negative throughout. News contribution is unstable
and folds-dependent even as an add-on to a working baseline.

## 4.5 Calibration (done) — served probabilities are honest

`scripts/validate_phase4_calibration.py`: out-of-fold predicted probabilities
vs labels over the same walk-forward (locked excluded). Brier 0.2025 pooled
vs 0.2500 constant train-rate baseline; per-asset 0.198–0.207.

Pooled decile reliability (mean predicted → empirical rate, ~5,470/bin):

```text
0.227 → 0.178 | 0.285 → 0.245 | 0.328 → 0.308 | 0.371 → 0.362
0.420 → 0.417 | 0.476 → 0.478 | 0.547 → 0.561 | 0.634 → 0.655
0.748 → 0.752 | 0.907 → 0.890
```

Tracks within ~1–5pp with mild low-end overconfidence. No recalibration
layer fitted — not warranted, and it would add overfit surface against the
scored-once locked discipline.

## 4.6 Google Trends attention on top of volatility baseline (done)

`scripts/fetch_google_trends.py` + `scripts/validate_phase4_trends_on_top.py`:
daily IN search interest for all 5 names (6-month chunks force daily rows;
1001 days each), relative features `interest[D-1]/median(past 30d)` and 7d
mean (robust to per-query rescaling; strictly pre-prediction). Same folds as
pilot, full history coverage — no sparsity excuse.

| Asset | Folds | Median BA delta | Win | Verdict |
|---|---|---:|---:|---|
| HDFCBANK | 26 | −0.21pp | 23% | FAIL |
| ICICIBANK | 26 | −0.11pp | 38% | FAIL |
| INFY | 26 | +0.08pp | 54% | FAIL |
| RELIANCE | 26 | −0.10pp | 50% | FAIL |
| TCS | 26 | −0.84pp | 31% | FAIL |
| POOLED | 130 | −0.19pp | 39% | FAIL |

Search attention adds nothing (slightly negative) over the market-only
volatility baseline at full statistical power.

## Phase 4 verdict

Volatility-regime prediction is the project's first robust signal (4.1–4.3:
130/130, minimal 9-feat set, 1h frozen; 4.5: calibrated). News does not add
stable incremental value for direction (Phases 0–3) **or** for volatility
(4.4). News work stops here unless a new representation is proposed.
