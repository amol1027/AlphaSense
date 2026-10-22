# Shadow Readouts — Prospective Volatility Prediction (Phase 4 → live)

The frozen 9-feature, 1h volatility model runs in prospective shadow mode: the
runner refreshes completed Upstox 15m candles into a rolling cache, logs
predictions from that cache only, and labels each prediction once the next
hour's bars exist. Scoring is read-only against the log; model, artifacts, and
the locked 2026-08-10+ holdout are never touched.

## Pipeline

- `scripts/run_shadow.py` — boundary-aligned scheduler; wakes just after each
  15m bar boundary (+120s completion delay) during NSE hours, refresh → label → log.
- `scripts/log_live_prediction.py` — log/label for `data/interim/live_log.csv`,
  keyed by (asset, prediction_timestamp), re-run safe. Realized labels use each
  bundle's frozen `range_median` (same threshold the server predicts against).
- `scripts/score_shadow.py` — stale-filtered per-asset cut with Wilson 95% CIs
  and balanced accuracy.

Logging gate (added after the first readout window): a prediction is logged
only when its 1h label window can complete inside the normal session — the
final session bar opens 15:15 IST, so predictions at 14:30–15:15 IST are
skipped (`label_window_completes()`). Rows logged before this gate exist in the
log as permanently unscored; they are excluded from scoring.

## Readout 1 — 2026-09-23 → 2026-10-21 (scored 2026-10-22)

Predictions span 10 sessions at uneven runner uptime (partially down early on
some days); 340/455 rows scored, 115 unscored are pre-gate dead slots, 0 stale.
Scored range 2026-09-28 10:00 → 2026-10-21 14:15 IST.

| Asset | n | Accuracy | Wilson 95% | BA | Majority base | BA delta |
|---|---:|---:|---|---:|---:|---:|
| RELIANCE | 68 | 0.721 | [0.604, 0.813] | 0.721 | 0.500 | +22.1pp |
| ICICIBANK | 68 | 0.676 | [0.558, 0.776] | 0.692 | 0.544 | +14.8pp |
| HDFCBANK | 68 | 0.662 | [0.543, 0.763] | 0.681 | 0.662 | +1.9pp |
| INFY | 68 | 0.632 | [0.514, 0.737] | 0.623 | 0.706 | −8.3pp |
| TCS | 68 | 0.706 | [0.589, 0.801] | 0.635 | 0.765 | −13.0pp |
| POOLED | 340 | 0.679 | [0.628, 0.727] | — | — | — |

Majority base = share of the modal realized label over the same scored rows; it
varies per asset with per-day label imbalance, and walks forward as the log
grows. Wilson intervals are on accuracy, not BA.

## Reading

- The historical walk-forward edge is **not uniform prospectively**. RELIANCE
  and ICICIBANK clearly beat their majority bases; HDFCBANK is flat; INFY and
  TCS sit below majority over the same months.
- This is the first prospective test of any signal in this project. An
  asset-dependent split after one month is not yet a verdict either way:
  n=68/asset, the majority base itself is noisy at this size, and the split
  pattern could be a volatility-regime coincidence of this particular month.
- No retraining, threshold changes, or artifact changes have occurred since
  Phase 4 froze; all conclusions in this file are out-of-sample by
  construction.

## Data hygiene (2026-10-22)

- An interim pass at n=290 under-counted rows from sessions where the runner
  was down; 50 recoverable rows were back-labeled after fetching the missed
  Upstox history (2026-09-28 → 2026-10-15). Only forward-window-complete rows
  were labeled; near-close slots have no label by construction.
- The logging gate now prevents future unlabelable rows.

## Next

- Keep the shadow run going; re-score periodically and append Readout 2+ here.
- No direction/news work; no threshold or artifact changes. The locked
  holdout stays scored-once.
- A prospective verdict needs materially more data before any per-asset
  enable/disable or deployment decision.
