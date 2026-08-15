# AlphaSense

**AlphaSense** is an NSE/BSE-focused research project for building a next-hour market-direction prediction pipeline using market data, financial news, and Reddit/social signals.

## Current Status

**Phase 0A — Foundation: COMPLETE**

**Current test status: 48 passed**

The foundation currently covers:

- NSE/BSE scope
- 1-hour prediction horizon
- Canonical market-data schema
- IST normalization
- NSE/BSE session validation
- Trading-calendar abstraction
- 2026 holiday reference data
- Calendar-aware feature construction
- News ingestion and sentiment aggregation
- Reddit ingestion and social features
- Next-hour target generation
- Temporal alignment and leakage prevention
- Automated tests

**Next milestone: Phase 0B.1 — Chronological train/validation/test split contract**

---

## Architecture

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

## Project Scope

### Market

The system is currently limited to:

- NSE
- BSE
- Normal Indian equity-market trading
- Asia/Kolkata (IST)

Normal continuous equity-session logic is based on **09:15–15:30 IST**.

Special/pre-open/post-close/auction-only sessions are not part of the initial normal-session prediction universe.

### Prediction Horizon

The current prediction task is:

> Predict whether the next trading hour will move up or down.

For a prediction timestamp `t`, the target compares the current close with the close one hour later.

The project is **not currently a next-day-only prediction system**.

---

## Temporal Contract

A feature at prediction time `t` may use only information available at or before `t`.

```text
Information available by t
          ↓
       Features
          ↓
 Predict t → t + 1 hour
```

For example, a TCS article published at 14:00 IST can be used for a 14:15 prediction. An article published at 14:30 cannot.

This rule applies to news, Reddit/social data, market-derived features, and every other model input.

---

## Market Session Contract

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

---

## Timezone Contract

Exchange-session logic uses:

```text
Asia/Kolkata
```

Python's standard `zoneinfo` implementation is used.

Naive timestamps are currently interpreted as IST. Timezone-aware timestamps are converted to IST before session validation.

This is important because external news/social data may arrive in UTC.

---

## Trading Calendar

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

## Market Data Schema

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

The downstream pipeline should not depend on whether the source is a CSV, API, broker, database, or exchange feed. External data should first be converted to this canonical representation.

---

## News Features

News is filtered using the prediction timestamp.

Current aggregated features include:

```text
sentiment_mean
sentiment_std
news_count
positive_ratio
negative_ratio
```

The development environment currently uses a dummy sentiment provider. The interface is designed so a production sentiment model such as FinBERT can later replace it without changing the downstream feature contract.

---

## Reddit / Social Features

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

## Target Definition

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

Rows without a complete future hour are excluded from the final feature dataset.

---

## Feature Dataset

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

## Leakage Prevention

Temporal leakage prevention is a primary design requirement.

The rule is:

> A feature for prediction timestamp `t` may only use information available at or before `t`.

Future market information is used only to construct the target, never as an input feature.

The next phase will extend this principle to the training/evaluation split.

---

## Testing

The project follows a test-driven development approach.

Tests currently cover:

- Feature aggregation
- Hourly sentiment
- News ingestion
- Reddit ingestion
- Sentiment provider behavior
- Target generation
- Time-window filtering
- Session windows
- IST conversion
- Trading calendar behavior
- Holiday data
- Holiday-aware feature construction
- End-to-end feature construction

Run the full suite with:

```powershell
pytest
```

Current result:

```text
48 passed
```

---

## Repository Structure

```text
AlphaSense/
│
├── data/
│   ├── raw/
│   │   ├── market_sample.csv
│   │   ├── news_sample.csv
│   │   ├── reddit_sample.csv
│   │   └── market_holiday_sample.csv
│   ├── processed/
│   └── reference/
│       └── nse_bse_holidays_2026.json
│
├── src/
│   ├── features/
│   │   ├── build_features.py
│   │   ├── market_time.py
│   │   ├── session_windows.py
│   │   ├── trading_calendar.py
│   │   ├── targets.py
│   │   ├── sentiment_features.py
│   │   └── reddit_features.py
│   ├── ingestion/
│   │   └── loader.py
│   └── sentiment/
│       └── dummy.py
│
├── tests/
│   └── ...
│
├── .gitignore
└── README.md
```

The repository structure will evolve as modeling and production-data work begins.

---

## Development Principles

### 1. Test before complexity

New behavior should be covered by tests.

### 2. Time first

Temporal correctness is more important than model complexity.

### 3. No look-ahead

Future information must never enter features used for an earlier prediction.

### 4. Explicit contracts

Market, time, calendar, sentiment, and target behavior should have clear interfaces and tests.

### 5. Replaceable providers

The downstream pipeline should not depend directly on one external provider.

### 6. Separate layers

Ingestion, feature engineering, modeling, prediction serving, and GUI remain separate concerns.

---

## Roadmap

### Phase 0A — Foundation

**Complete**

```text
Data contracts
Temporal alignment
Session/calendar handling
Feature construction
Targets
Automated tests
```

### Phase 0B — Modeling & Evaluation

**Next**

1. Define chronological train/validation/test split.
2. Prevent cross-period leakage.
3. Establish simple baseline models.
4. Define evaluation metrics.
5. Build a reproducible training pipeline.
6. Compare model performance against simple baselines.

### Later

```text
Real NSE/BSE market data
        ↓
Real financial news
        ↓
Real social data
        ↓
Production sentiment model
        ↓
Feature dataset
        ↓
ML models
        ↓
Prediction API
        ↓
GUI
```

---

## Current Limitations

This is still a research/development prototype.

- Market data is currently sample/development data.
- News data is currently sample/development data.
- Reddit data is currently sample/development data.
- Sentiment currently uses a dummy provider.
- The ML model has not yet been trained.
- No live trading system exists.
- No investment recommendation should be inferred from current outputs.
- The holiday reference data must be maintained when official exchange calendars are updated.

---

## Current Milestone

```text
Phase 0A — Foundation
STATUS: COMPLETE

Tests:
48 passed

Next:
Phase 0B.1 — Chronological train/validation/test split contract
```
