# AlphaSense

**AlphaSense** is an NSE/BSE-focused research project for building a next-hour market-direction prediction pipeline using market data, financial news, and Reddit/social signals.

## Current Status

**Phase 0A — Foundation: COMPLETE**

**Phase 0B — Modeling & Evaluation: IN PROGRESS**

**Current test status: 190 passed, 1 warning**

The project currently covers:

* NSE/BSE scope
* 1-hour prediction horizon
* Canonical market-data schema
* IST normalization
* NSE/BSE session validation
* Trading-calendar abstraction
* 2026 holiday reference data
* Calendar-aware feature construction
* Historical financial-news ingestion
* FinBERT sentiment inference
* News sentiment aggregation
* Reddit ingestion and social features
* Next-hour target generation
* Temporal alignment and leakage prevention
* Chronological train/validation/test splitting
* Logistic Regression baseline
* Feature ablation
* Asset-level evaluation
* Expanding-window walk-forward evaluation
* News-vs-market per-period comparison
* Automated tests

### Current research finding

The current Logistic Regression baseline does **not** demonstrate robust within-asset predictive power.

The current FinBERT news features also do **not** demonstrate a stable predictive improvement over the market-only baseline.

This is a research finding, not a failure of the ingestion or sentiment pipeline.

**Next milestone: Normalize market features and establish stronger majority/market/news baselines.**

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

The current real feature dataset contains:

```text
1,276 feature rows
638 RELIANCE
638 TCS
29 trading days
22 prediction rows per asset per trading day
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

The current baseline model is:

```text
Logistic Regression
```

Current experiments compare:

```text
Market only

Market + News

Market + Reddit

Market + News + Reddit
```

The current market feature set uses:

```text
open
high
low
close
volume
```

These are currently raw price/volume features.

**Next modeling improvement:** replace raw price levels with normalized, scale-independent market features.

Candidate features include:

```text
return_15m
return_30m
return_1h
high_low_range
close_open_return
volume_change
volume_zscore
```

These will be evaluated without removing the existing raw-feature baseline, allowing a direct comparison.

---

# Current Walk-Forward Results

The current expanding-window experiment uses:

```text
29 trading days
10 initial training days
19 out-of-sample test periods
```

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

The next evaluation stage will establish a stronger hierarchy of baselines:

```text
Majority-class baseline
        ↓
Raw market Logistic Regression
        ↓
Normalized market Logistic Regression
        ↓
Normalized market + News
        ↓
Normalized market + News + Reddit
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
190 passed, 1 warning
```

The warning currently comes from the tokenizer dependency used by the FinBERT stack and does not fail the test suite.

---

# Repository Structure

```text
AlphaSense/
│
├── data/
│   ├── raw/
│   │   ├── market/
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
│   ├── collect_historical_news.py
│   ├── process_historical_news_sentiment.py
│   ├── evaluate_feature_ablation.py
│   ├── evaluate_asset_ablation.py
│   ├── evaluate_real_logistic.py
│   └── evaluate_walk_forward.py
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

**IN PROGRESS**

Completed:

```text
Chronological train/validation/test split
Leakage-safe modeling dataset
Logistic Regression baseline
Feature ablation
Asset-level evaluation
Walk-forward evaluation
News-vs-market comparison
```

Next:

```text
Majority-class baseline
        ↓
Normalized market features
        ↓
Normalized market baseline
        ↓
Normalized market + News
        ↓
Robust walk-forward comparison
        ↓
Statistical/robustness analysis
```

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
* The current model is a simple Logistic Regression baseline.
* Raw OHLCV price levels are still being used as the market feature baseline.
* Walk-forward evaluation currently contains 19 test periods.
* Statistical conclusions remain preliminary.
* No live trading system exists.
* No investment recommendation should be inferred from current outputs.
* The holiday reference data must be maintained when official exchange calendars are updated.

---

# Current Milestone

```text
Phase 0A — Foundation
STATUS: COMPLETE

Phase 0B — Modeling & Evaluation
STATUS: IN PROGRESS

Tests:
190 passed, 1 warning

Current research status:
- Leakage-safe chronological evaluation is implemented.
- Raw-market Logistic Regression is near-chance within individual assets.
- Pooled market-only performance is modestly above chance but unstable.
- Current FinBERT News features do not show a stable improvement.
- Reddit features currently show no measurable improvement.

Next:
Normalize market features and establish
majority / market / market + news baselines.
```
