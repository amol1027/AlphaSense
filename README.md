# AlphaSense

**AlphaSense** is an NSE/BSE-focused research project for predicting short-horizon market behavior using market data, financial news, and Reddit/social signals. Next-hour *direction* research is frozen as negative; the active signal is next-hour *range-regime* (volatility) prediction.

## Current Status

**Phase 0 — Small-sample direction research: COMPLETE (see `docs/phase_0_results.md`)**

**Phase 1 — Expanded direction dataset, frozen locked evaluation: FROZEN (see `docs/phase_1_results.md`)**

**Phase 2 — News-signal consolidation through 2.7: CLOSED (see `docs/phase_2_consolidation.md`)**

**Phase 3 — Conditional-signal validation + 5-asset market baselines: COMPLETE (see `docs/phase_3_results.md`)**

**Phase 4 — Volatility-regime pilot: PASS (see `docs/phase_4_pilot.md`)**

**Current test status: 256 passed, 2 warnings**

The project currently covers:

* NSE/BSE scope
* 1-hour prediction horizon
* Canonical market-data schema
* IST normalization
* NSE/BSE session validation
* Trading-calendar abstraction
* 2024–2026 NSE/BSE holiday reference data
* Special-session calendar handling
* Calendar-aware feature construction
* Historical financial-news ingestion
* FinBERT sentiment inference
* News sentiment aggregation
* Reddit ingestion and social features
* Next-hour target generation (binary + Phase 2 three-class thresholded ±0.203666%)
* Temporal alignment and leakage prevention
* Chronological train/validation/test splitting
* Logistic Regression baseline + Random Forest / HistGradientBoosting comparison
* Normalized/scale-independent market features (`src/features/market_features.py`)
* Majority-class baseline (`src/modeling/baseline.py`)
* Feature ablation
* Asset-level evaluation
* Expanding-window walk-forward evaluation
* News-vs-market per-period and matched-sample comparison
* Phase 1 locked holdout evaluation (2026-08-10 onward, scored once, never tuned against)
* 5-asset market coverage: RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK (83,259 combined 15m rows)
* Next-hour range-regime (volatility) prediction — the first robust signal (Phase 4 PASS, 130/130 folds)
* Frozen 9-feat volatility set: normalized market features + 1h trailing range/return stats
* Automated tests

### Current research finding

Direction prediction is frozen as negative: on 5 assets / 130 walk-forward folds the normalized-feature logistic has median balanced-accuracy delta of +0.1pp over majority (55% win rate) and loses to majority on the locked holdout for every asset. News adds no stable value for direction (Phases 0–3) or on top of the volatility baseline (Phase 4.4).

Range-regime (volatility) prediction passes robustly: 130/130 folds won, median +13.9pp, locked holdout beaten on all 5 assets (up to 0.68 vs 0.50 baseline). Edge decays with horizon (30m ≈ 1h > 2h > 4h); 1h frozen. See `docs/phase_3_results.md` and `docs/phase_4_pilot.md`.

**Next milestone: run the frozen 9-feature, 1h volatility model in prospective shadow mode. Refresh completed 15m Upstox candles into `data/interim/live_market_15m.csv`, serve and log predictions from that cache, then label them only after the next hour is present. No further direction/news iteration without a new hypothesis; locked holdout stays scored-once.**

To start the local dashboard and shadow runner, see [docs/STARTING_THE_APP.md](docs/STARTING_THE_APP.md).

---

# Architecture

```text
NSE/BSE Market Data ─┐
Financial News ──────┤
Reddit/Social ───────┤
                     ↓
              Time Alignment
                     ↓
              Feature Pipeline
                     ↓
             Next-Hour Target
                     ↓
               ML Model Layer
                     ↓
              Prediction API
                     ↓
                    GUI
```

The GUI is intentionally a later layer. It should consume predictions/features through a separate interface rather than contain prediction logic.

---

# Project Scope

## Market

The system is currently limited to:

* NSE
* BSE
* Normal Indian equity-market trading
* Asia/Kolkata (IST)

Normal continuous equity-session logic is based on **09:15–15:30 IST**.

Special/pre-open/post-close/auction-only sessions are not part of the initial normal-session prediction universe.

---

## Prediction Horizon

The current prediction task is:

> Predict whether the next trading hour will move up or down.

For a prediction timestamp `t`, the target compares the current close with the close one hour later.

The project is **not currently a next-day-only prediction system**.

---

# Temporal Contract

A feature at prediction time `t` may use only information available at or before `t`.

```text
Information available by t
          ↓
       Features
          ↓
 Predict t → t + 1 hour
```

For example, a TCS article published at 14:00 IST can be used for a 14:15 prediction. An article published at 14:30 cannot.

This rule applies to:

* news
* Reddit/social data
* market-derived features
* every other model input

Future market information is used only to construct the target.

---

# Market Session Contract

A prediction timestamp must:

1. Fall on a trading day.
2. Fall inside the normal NSE/BSE continuous equity session.
3. Have a complete one-hour future window.

Therefore:

```text
09:15 → 10:15   valid
10:15 → 11:15   valid
...
14:15 → 15:15   valid
15:15 → 16:15   invalid
```

The final incomplete session window is excluded.

The Phase 0 real feature dataset contained:

```text
1,276 feature rows
638 RELIANCE
638 TCS
29 trading days
22 prediction rows per asset per trading day
```

The expanded Phase 1 feature dataset (`data/processed/phase1_features.csv`, git-ignored build artifact) contains:

```text
28,564 feature rows
14,282 RELIANCE
14,282 TCS
2024-01-01 → 2026-08-21
Final locked holdout from 2026-08-10 (10 sessions, 210 usable obs/asset)
```

---

# Timezone Contract

Exchange-session logic uses:

```text
Asia/Kolkata
```

Python's standard `zoneinfo` implementation is used.

Naive timestamps are currently interpreted as IST. Timezone-aware timestamps are converted to IST before session validation.

This is important because external news/social data may arrive in UTC.

---

# Trading Calendar

The project uses a replaceable calendar abstraction:

```text
TradingCalendar
      │
      ├── WeekdayTradingCalendar
      │
      └── NSEBSETradingCalendar
```

The production-oriented calendar loads holiday data from:

```text
data/reference/nse_bse_holidays_2024.json
data/reference/nse_bse_holidays_2025.json
data/reference/nse_bse_holidays_2026.json
```

The calendar is kept separate from session logic so holiday data can be updated without rewriting prediction-window logic.

The feature builder also supports calendar injection for tests.

---

# Market Data Schema

Market records use the canonical `MarketBar` schema:

```text
MarketBar
├── asset
├── exchange
├── timestamp
├── open
├── high
├── low
├── close
└── volume
```

The downstream pipeline should not depend on whether the source is a CSV, API, broker, database, or exchange feed.

External data should first be converted to this canonical representation.

---

# Phase 1 Market Data Validation

The Phase 1 15-minute market dataset for RELIANCE and TCS has passed the
market-data audit.

```text
RELIANCE: PASS
TCS:      PASS

Rows per asset:          16,322
Observed sessions:         656
Expected sessions:         656
Missing 15m bars:             0
Unexpected 15m bars:          0
Duplicate timestamps:        0
Misaligned timestamps:        0
Weekend bars:                 0
Holiday bars:                 0
Invalid numeric values:       0
OHLC violations:              0
```

Cross-asset coverage is identical:

```text
Common timestamps: 16,322
RELIANCE-only:          0
TCS-only:               0
```

The audit is implemented in `src/ingestion/market/audit.py` and can be run with:

```powershell
python -m scripts.audit_phase1_market
```

---

# News Pipeline

Historical research news is collected from available news providers and stored as raw articles.

The current historical dataset contains:

```text
1,477 articles
RELIANCE: 1,218
TCS:       259
```

The historical news range is:

```text
2026-07-05 → 2026-08-10
```

Historical collection now uses the canonical `deduplicate_news()` implementation from `src/ingestion/news/dedup.py` before downstream processing.

Articles are deduplicated before being used downstream.

---

# News Preparation

News preparation follows the rule:

```text
article body available
        ↓
      use body

body unavailable
        ↓
    use headline
```

Articles with neither usable body text nor headline text are discarded.

This allows providers with incomplete article bodies to remain usable while maintaining a valid sentiment-input contract.

---

# News Sentiment

News sentiment is currently generated using **FinBERT**.

The sentiment pipeline produces article-level:

```text
sentiment_score
positive_probability
neutral_probability
negative_probability
```

The current aggregated news features are:

```text
sentiment_mean
sentiment_std
news_count
positive_ratio
negative_ratio
```

News is associated with a prediction timestamp only when:

```text
published_at <= prediction_timestamp
```

News exchange metadata is not used as the primary join key; asset identity is used for association.

---

# Reddit / Social Features

Reddit is aligned using the same temporal cutoff.

Current features include:

```text
reddit_sentiment_mean
reddit_sentiment_std
reddit_count
reddit_positive_ratio
reddit_negative_ratio
reddit_score_mean
reddit_comments_mean
reddit_engagement_mean
```

Current development engagement is represented as:

```text
engagement = score + comments
```

Social signals are treated as an additional modality.

---

# Target Definition

For a market row at time `t`:

```text
current close = close(t)
future close  = close(t + 1 hour)
```

The pipeline produces:

```text
future_close
target_return
target_direction
```

The current directional target is:

```text
future return > 0  → 1 (UP)
future return <= 0 → 0 (DOWN)
```

If no observation exists exactly one hour later, the target is missing.

Rows without a complete future hour are excluded from the final modeling dataset.

---

# Feature Dataset

The combined feature dataset currently contains market, news, Reddit, and target fields such as:

```text
asset
exchange
prediction_timestamp

open
high
low
close
volume

future_timestamp
future_close
target_return
target_direction

sentiment_mean
sentiment_std
news_count
positive_ratio
negative_ratio

reddit_sentiment_mean
reddit_sentiment_std
reddit_count
reddit_positive_ratio
reddit_negative_ratio
reddit_score_mean
reddit_comments_mean
reddit_engagement_mean
```

---

# Leakage Prevention

Temporal leakage prevention is a primary design requirement.

The rule is:

> A feature for prediction timestamp `t` may only use information available at or before `t`.

The current implementation applies this rule to:

* news
* Reddit
* sentiment
* market-derived features

Future market information is used only for:

```text
future_close
target_return
target_direction
```

and is never included as a model feature.

Chronological train/validation/test splitting is implemented.

Expanding-window walk-forward evaluation is also implemented:

```text
Train ─────────→ Test

Train ─────────────→ Test

Train ─────────────────→ Test
```

Each test period is evaluated using only information available before that period.

---

# Modeling

Baseline models evaluated:

```text
Logistic Regression
Random Forest
HistGradientBoosting
Majority-class baseline (src/modeling/baseline.py)
```

Current experiments compare:

```text
Market only

Market + News

Market + Reddit

Market + News + Reddit
```

The frozen market feature set uses normalized features (`src/features/market_features.py`):

```text
return_15m
return_30m
return_1h
high_low_range
close_open_return
volume_change
```

Raw OHLCV levels are retained only as the Phase 0 baseline for comparison. Phase 1 frozen candidates are documented in `docs/phase_1_results.md`.

---

# Phase 0 Walk-Forward Results (small-sample, superseded)

The expanding-window experiment below used 29 trading days / 10 initial training days / 19 OOS periods. It is retained for history only. Locked results are in `docs/phase_0_results.md`, `docs/phase_1_results.md`.

## All Assets

| Experiment             | Mean Accuracy | Mean Balanced Accuracy |
| ---------------------- | ------------: | ---------------------: |
| Market only            |        53.38% |             **53.95%** |
| Market + News          |        52.76% |                 52.96% |
| Market + Reddit        |        53.38% |             **53.95%** |
| Market + News + Reddit |        52.76% |                 52.96% |

Reddit currently produces no measurable difference.

News reduces mean balanced accuracy by approximately:

```text
53.95% → 52.96%
Δ = -0.99 percentage points
```

---

## RELIANCE

| Experiment             | Mean Balanced Accuracy |
| ---------------------- | ---------------------: |
| Market only            |             **50.64%** |
| Market + News          |                 47.54% |
| Market + Reddit        |             **50.64%** |
| Market + News + Reddit |                 47.54% |

The current RELIANCE-specific market model is approximately chance-level.

News currently reduces performance by approximately:

```text
-3.10 percentage points
```

---

## TCS

| Experiment             | Mean Balanced Accuracy |
| ---------------------- | ---------------------: |
| Market only            |                 49.97% |
| Market + News          |             **50.38%** |
| Market + Reddit        |                 49.97% |
| Market + News + Reddit |             **50.38%** |

The News improvement is only approximately:

```text
+0.41 percentage points
```

which is currently too small to treat as meaningful evidence of predictive value.

---

# News vs Market Per-Period Analysis

Across the 19 pooled walk-forward test periods:

```text
News wins:  6
News loses: 6
Ties:       7
```

Therefore, News is currently neither consistently helpful nor consistently harmful.

For RELIANCE:

```text
News wins:  4
News loses: 6
Ties:       9
```

For TCS:

```text
News wins:  6
News loses: 4
Ties:       9
```

The per-period results show substantial variation between test periods.

This reinforces the need for stronger market features and additional robustness testing before making conclusions about the usefulness of financial-news sentiment.

---

# Current Research Interpretation

The current evidence supports the following conclusion:

> The existing FinBERT news representation combined with the current Logistic Regression model has not demonstrated a stable improvement in next-hour directional prediction.

This does **not** establish that financial news has no predictive information.

Possible explanations still include:

* raw market features are poorly normalized
* the model is too simple
* the sample period is small
* the news representation may not capture event timing effectively
* article coverage is incomplete
* asset-specific behavior differs
* the next-hour horizon may contain weak/noisy directional signal
* the current classification formulation may discard useful return magnitude information

The research pipeline therefore remains active.

---

# Baselines

Implemented baseline hierarchy (`src/modeling/baseline.py`, `src/modeling/baseline_runner.py`):

```text
Majority-class baseline
        ↓
Raw market Logistic Regression (Phase 0)
        ↓
Normalized market Logistic Regression / tree models
        ↓
Normalized market + News
        ↓
Normalized market + News + Reddit (sample-only)
```

All comparisons should use the same chronological/walk-forward evaluation framework.

---

# Testing

The project follows a test-driven development approach.

Tests currently cover:

* Feature aggregation
* Hourly sentiment
* News ingestion
* Reddit ingestion
* Sentiment provider behavior
* FinBERT behavior
* Target generation
* Time-window filtering
* Session windows
* IST conversion
* Trading calendar behavior
* Holiday data
* Holiday-aware feature construction
* End-to-end feature construction
* Modeling dataset construction
* Chronological splits
* Logistic Regression behavior
* News/Reddit feature integration
* Sentiment preparation
* News deduplication
* Walk-forward-related modeling behavior

Run the full suite with:

```powershell
pytest
```

Current result:

```text
256 passed, 2 warnings
```

Warnings:
1. `transformers` BERT tokenizer `WordPiece.__init__` deprecation (FinBERT stack, does not fail suite).
2. `src/ingestion/news/audit.py` dateutil fallback parse warning in `test_news_audit.py` (does not fail suite).

---

# Repository Structure

```text
AlphaSense/
│
├── data/
│   ├── raw/
│   │   ├── market/
│   │   │   ├── tcs_15m.csv
│   │   │   ├── reliance_15m.csv
│   │   │   └── research_market_15m.csv
│   │   ├── news/
│   │   │   └── research_news.csv
│   │   ├── reddit_sample.csv
│   │   └── market_holiday_sample.csv
│   │
│   ├── processed/
│   │   └── research_news_sentiment.csv
│   │
│   └── reference/
│       └── nse_bse_holidays_2026.json
│
├── scripts/
│   │
│   ├── Data collection
│   │   ├── download_market_data.py
│   │   ├── collect_historical_news.py
│   │   └── build_research_market.py
│   │
│   ├── Sentiment processing
│   │   └── process_historical_news_sentiment.py
│   │
│   ├── Evaluation
│   │   ├── evaluate_real_logistic.py
│   │   ├── evaluate_real_baseline.py
│   │   ├── evaluate_feature_ablation.py
│   │   ├── evaluate_asset_ablation.py
│   │   ├── evaluate_logistic_by_asset.py
│   │   ├── evaluate_logistic_per_asset.py
│   │   ├── evaluate_baseline.py
│   │   └── evaluate_walk_forward.py
│   │
│   └── Diagnostics
│       ├── inspect_real_targets.py
│       ├── check_news_overlap.py
│       ├── check_upstox_news_api.py
│       ├── resolve_instruments.py
│       └── test_historical_request.py
│
├── src/
│   ├── features/
│   │   ├── build_features.py
│   │   ├── market_time.py
│   │   ├── session_windows.py
│   │   ├── trading_calendar.py
│   │   ├── targets.py
│   │   ├── sentiment_features.py
│   │   ├── reddit_features.py
│   │   └── time_windows.py
│   │
│   ├── ingestion/
│   │   ├── loader.py
│   │   ├── reddit_loader.py
│   │   └── news/
│   │
│   ├── sentiment/
│   │   ├── finbert.py
│   │   ├── pipeline.py
│   │   ├── news_preparation.py
│   │   ├── hourly_aggregation.py
│   │   └── schemas.py
│   │
│   └── modeling/
│       ├── dataset.py
│       ├── splits.py
│       ├── logistic.py
│       ├── baseline.py
│       ├── baseline_runner.py
│       ├── benchmark.py
│       └── evaluation.py
│
├── tests/
│   └── ...
│
├── .gitignore
└── README.md
```

The repository structure will continue to evolve as modeling and production-data work progresses.

---

# Scripts

All scripts must be run from the project root directory.

```powershell
python scripts/<script_name>.py
```

---

## Data Collection

### `download_market_data.py`

Downloads 15-minute OHLCV candles for TCS and RELIANCE from Upstox over a
configured date range and saves them as per-asset CSV files.

```text
Output:
  data/raw/market/tcs_15m.csv
  data/raw/market/reliance_15m.csv
```

Requires a valid Upstox API key in `.env`.

---

### `build_research_market.py`

Merges the per-asset market CSVs into a single combined file sorted by asset
and timestamp. Run this after `download_market_data.py`.

```text
Input:
  data/raw/market/tcs_15m.csv
  data/raw/market/reliance_15m.csv

Output:
  data/raw/market/research_market_15m.csv
```

---

### `collect_historical_news.py`

Collects historical financial news for TCS and RELIANCE over a configured date
range using both Marketaux and GDELT as sources. Uses the canonical
`deduplicate_news()` implementation from `src/ingestion/news/dedup.py` before
writing the output.

GDELT windows that hit the 250-record limit are automatically bisected to
avoid silent data loss.

```text
Output:
  data/raw/news/research_news.csv
```

Requires a Marketaux API key in `.env`. GDELT is free but rate-limited
(5-second sleep between requests).

---

## Sentiment Processing

### `process_historical_news_sentiment.py`

Loads the raw news CSV, prepares article text for FinBERT (body preferred,
fallback to headline), runs FinBERT sentiment inference, and writes
article-level sentiment scores.

```text
Input:
  data/raw/news/research_news.csv

Output:
  data/processed/research_news_sentiment.csv

Columns written:
  asset, exchange, published_at, source, text,
  positive_probability, neutral_probability,
  negative_probability, sentiment_score
```

This script is slow (FinBERT is run locally). Run it once and reuse
`research_news_sentiment.csv` in all subsequent evaluation scripts.

---

## Evaluation

All evaluation scripts use the same data paths:

```text
Market:    data/raw/market/research_market_15m.csv
News:      data/raw/news/research_news.csv
Sentiment: data/processed/research_news_sentiment.csv
Reddit:    data/raw/reddit_sample.csv
```

---

### `evaluate_walk_forward.py`

**Primary evaluation script.** Runs an expanding-window walk-forward experiment
across all assets and per asset (RELIANCE, TCS) for four experiment groups:

```text
Market only
Market + News
Market + Reddit
Market + News + Reddit
```

Each test period is one trading day. The model trains on all prior days and
predicts that day's observations. The minimum initial training window is
10 trading days.

Also runs a news-vs-market per-period comparison and reports win/loss/tie
counts.

```powershell
python scripts/evaluate_walk_forward.py
```

---

### `evaluate_feature_ablation.py`

Runs the same four experiment groups using a fixed chronological
train/validation/test split instead of walk-forward.

```text
Train end:      2026-07-25
Validation end: 2026-08-02
```

Reports validation and test accuracy, balanced accuracy, precision, and recall
for each experiment.

```powershell
python scripts/evaluate_feature_ablation.py
```

---

### `evaluate_asset_ablation.py`

Runs the feature ablation evaluation broken down by individual asset using the
same fixed train/validation/test split as `evaluate_feature_ablation.py`.

```powershell
python scripts/evaluate_asset_ablation.py
```

---

### `evaluate_real_logistic.py`

Fits and evaluates a Logistic Regression model on the real research dataset
using a fixed chronological split. Reports validation and test metrics for
market-only and market-plus-news feature sets.

```powershell
python scripts/evaluate_real_logistic.py
```

---

### `evaluate_real_baseline.py`

Evaluates the majority-class baseline on the real dataset using a fixed
chronological split. Use this to establish the floor accuracy that any model
must beat.

```powershell
python scripts/evaluate_real_baseline.py
```

---

### `evaluate_logistic_by_asset.py`

Fits and evaluates a Logistic Regression model separately for each asset.
Allows per-asset performance to be compared directly.

```powershell
python scripts/evaluate_logistic_by_asset.py
```

---

### `evaluate_logistic_per_asset.py`

Variant of asset-level evaluation with additional per-period breakdown output.

```powershell
python scripts/evaluate_logistic_per_asset.py
```

---

### `evaluate_baseline.py`

Evaluates the majority-class baseline and random baseline on development sample
data. Useful for sanity-checking the baseline infrastructure independently of
the real dataset.

```powershell
python scripts/evaluate_baseline.py
```

---

## Diagnostics

### `inspect_real_targets.py`

Loads the per-asset 15-minute market CSVs, computes next-hour targets, and
prints class distribution and a sample of target rows. Use this to verify that
the target generation is working correctly before running model evaluation.

```powershell
python scripts/inspect_real_targets.py
```

---

### `check_news_overlap.py`

Fetches a small sample of recent articles from both Marketaux and Upstox for
TCS and RELIANCE, deduplicates them, and prints the overlap statistics. Use
this to assess how much duplicate coverage exists across news providers.

Requires API keys for both providers in `.env`.

```powershell
python scripts/check_news_overlap.py
```

---

### `check_upstox_news_api.py`

Sanity-checks the Upstox news API connection and prints a small sample of
recently fetched articles. Use this to confirm that the API key and client
behavior are working correctly.

```powershell
python scripts/check_upstox_news_api.py
```

---

### `resolve_instruments.py`

Looks up Upstox instrument keys for a configured list of symbols. Use this
when adding new assets to confirm the correct `instrument_key` before
updating `download_market_data.py`.

```powershell
python scripts/resolve_instruments.py
```

---

### `test_historical_request.py`

Sends a single historical candle request to the Upstox API and prints the raw
response. Use this to debug API authentication or date-range issues in
isolation before running the full downloader.

```powershell
python scripts/test_historical_request.py
```

### `check_upstox_token.py`

Fail-loud token gate (exit 0 = valid, 1 = expired/missing). Upstox access
tokens expire daily and require manual OAuth login to renew — there is no
programmatic refresh, so run this before any refresh job or scheduled task
rather than serving stale bars.

```powershell
python scripts/check_upstox_token.py
```

---

# Development Principles

### 1. Test before complexity

New behavior should be covered by tests.

### 2. Time first

Temporal correctness is more important than model complexity.

### 3. No look-ahead

Future information must never enter features used for an earlier prediction.

### 4. Explicit contracts

Market, time, calendar, sentiment, target, and modeling behavior should have clear interfaces and tests.

### 5. Replaceable providers

The downstream pipeline should not depend directly on one external provider.

### 6. Separate layers

Ingestion, feature engineering, modeling, prediction serving, and GUI remain separate concerns.

### 7. Evaluate before optimizing

Model changes should be justified by reproducible out-of-sample evaluation rather than a single favorable split.

### 8. Preserve baselines

Existing experiments should remain reproducible so new features/models can be compared against previous results.

---

# Roadmap

## Phase 0A — Foundation

**COMPLETE**

```text
Data contracts
Temporal alignment
Session/calendar handling
Feature construction
Targets
News ingestion
Sentiment pipeline
Reddit features
Automated tests
```

---

## Phase 0B — Modeling & Evaluation

**COMPLETE** — see `docs/phase_0_results.md`.

```text
Chronological train/validation/test split
Leakage-safe modeling dataset
Logistic Regression baseline + tree-model comparison
Majority-class baseline
Feature ablation
Asset-level evaluation
Walk-forward evaluation
News-vs-market comparison
Locked final evaluation (2026-08-03 onward, +0.96pp RELIANCE / +0.68pp TCS, not robust)
```

## Phase 1 — Expanded Dataset

**FROZEN** — see `docs/phase_1_results.md`. Do not change final test period, targets, candidates, hyperparameters, thresholds, or data processing.

## Phase 2 — News-Signal Consolidation

**CLOSED** — see `docs/phase2_news_signal_consolidation.md` (through 2.7, 60m window frozen, ±0.203666% threshold, locked holdout from 2026-08-10 protected).

---

## Later

```text
Real NSE/BSE market data
        ↓
Real financial news
        ↓
Real social data
        ↓
Production sentiment pipeline
        ↓
Normalized feature dataset
        ↓
ML models
        ↓
Model comparison
        ↓
Prediction API
        ↓
GUI
```

---

# Current Limitations

This is still a research/development prototype.

* The historical market dataset covers a limited research period.
* Historical news coverage is incomplete across providers.
* Some news articles do not contain full body text.
* Reddit data is currently limited development data.
* FinBERT is currently run locally.
* The current model set is Logistic Regression / Random Forest / HistGradientBoosting baselines (no production model).
* Normalized market features are now the frozen candidate representation; raw OHLCV is history-only.
* Walk-forward evaluation history includes the Phase 0 19-period experiment; Phase 1 uses locked holdout + daily breakdown.
* Statistical conclusions remain preliminary.
* No live trading system exists.
* No investment recommendation should be inferred from current outputs.
* The holiday reference data must be maintained when official exchange calendars are updated.

---

# Current Milestone

```text
Phase 0 — COMPLETE (direction, small sample, negative)

Phase 1 — FROZEN (direction, locked evaluation below baseline, audits PASS)

Phase 2 — CLOSED (news-signal consolidation; only conditional
high-sentiment-magnitude lead, small-N)

Phase 3 — COMPLETE (magnitude-regime FAIL; 5-asset expansion: market done,
news blocked on coverage; market-only 5-asset baselines FAIL, 130 folds)

Phase 4 — PASS (range-regime volatility pilot: 130/130 folds, minimal 9-feat
set frozen, 1h horizon frozen; news-on-top FAIL)

Tests:
256 passed, 2 warnings

Next:
Run the frozen 9-feature, 1h volatility model in prospective shadow mode.
`python scripts/run_shadow.py` refreshes completed Upstox 15m candles into a
rolling 7-day cache, labels outcomes once the next hour is present, and logs
fresh predictions during NSE hours. Original research market files are kept.
Run `python -m src.prediction.server` separately to serve the local dashboard/
API; it reads the refresher's cache. The refresh uses
`UPSTOX_ACCESS_TOKEN`; historical seed bars are excluded from prediction logs
by the freshness gate. No direction/news re-tuning; locked holdout stays
scored-once. Commits pending owner review (working tree intentionally uncommitted).
```
