from pathlib import Path

import pandas as pd


OUTPUT_PATH = Path(
    "docs/phase2_news_signal_consolidation.md"
)


def main() -> None:

    output = """# Phase 2 News Signal Consolidation

## Status

Phase 2 news-signal investigation completed through Phase 2.7.

The locked holdout beginning 2026-08-10 was not used for model fitting,
feature selection, threshold selection, or metric comparison.

---

## Frozen Experimental Configuration

| Item | Configuration |
|---|---|
| Prediction horizon | 1 hour |
| Target | Three-class thresholded return |
| Target threshold | ±0.203666% |
| News information window | Trailing 60 minutes |
| Development news start | 2026-07-05 |
| OOS start | 2026-07-25 |
| Locked holdout start | 2026-08-10 |

---

## Phase 2.3A — News Incremental Signal

### RELIANCE

| Metric | Market-only | Market + News | Change |
|---|---:|---:|---:|
| Accuracy | 0.5952 | 0.6741 | +0.0788 |
| Balanced accuracy | 0.3281 | 0.3812 | +0.0531 |
| Macro-F1 | 0.2488 | 0.3536 | +0.1049 |

### TCS

| Metric | Market-only | Market + News | Change |
|---|---:|---:|---:|
| Accuracy | 0.3857 | 0.2778 | -0.1079 |
| Balanced accuracy | 0.3614 | 0.2662 | -0.0952 |
| Macro-F1 | 0.3002 | 0.2339 | -0.0663 |

### Interpretation

The initial result suggested an asset-dependent news effect, with RELIANCE
appearing more promising than TCS.

However, the two models were evaluated on different observations because
rows without news features were excluded.

Therefore this result was not sufficient to establish incremental news value.

---

## Phase 2.3B — Matched-Sample News Incremental Signal

### RELIANCE

| Metric | Market-only | Market + News | Change |
|---|---:|---:|---:|
| Accuracy | 0.6593 | 0.6741 | +0.0148 |
| Balanced accuracy | 0.3260 | 0.3812 | +0.0552 |
| Macro-F1 | 0.2649 | 0.3536 | +0.0887 |

Matched sample:

- Training: 207
- OOS: 135

### TCS

| Metric | Market-only | Market + News | Change |
|---|---:|---:|---:|
| Accuracy | 0.3056 | 0.2778 | -0.0278 |
| Balanced accuracy | 0.2771 | 0.2662 | -0.0109 |
| Macro-F1 | 0.2222 | 0.2339 | +0.0117 |

Matched sample:

- Training: 129
- OOS: 72

### Interpretation

The matched-sample comparison provides stronger evidence than Phase 2.3A.

RELIANCE shows preliminary incremental value, particularly in balanced
accuracy and Macro-F1.

TCS does not show a consistent improvement.

The sample sizes remain small, so the result is exploratory.

---

## Phase 2.4 — News Window Investigation

The trailing 60-minute window was retained.

For RELIANCE, the 60-minute configuration produced the strongest incremental
result among the tested windows:

| Window | Accuracy Δ | Balanced accuracy Δ | Macro-F1 Δ |
|---|---:|---:|---:|
| 30m | -0.0093 | +0.0186 | +0.0402 |
| 60m | +0.0148 | +0.0552 | +0.0887 |
| 120m | -0.0280 | +0.0012 | +0.0192 |
| 240m | +0.0069 | +0.0034 | +0.0017 |

The 60-minute window was therefore frozen for subsequent experiments.

---

## Phase 2.6 — News Event Intensity

News intensity features were evaluated as an additive feature block.

### RELIANCE

| Metric change | Result |
|---|---:|
| Accuracy | +0.0492 |
| Balanced accuracy | +0.0065 |
| Macro-F1 | +0.0409 |

### TCS

| Metric change | Result |
|---|---:|
| Accuracy | -0.0976 |
| Balanced accuracy | -0.0814 |
| Macro-F1 | -0.0753 |

### Decision

News intensity was **not promoted** as a general feature block.

The RELIANCE balanced-accuracy improvement was too small to establish a
robust directional contribution, while TCS deteriorated across all primary
metrics.

---

## Phase 2.7 — Selective News / Event Regime

A selective event-regime approach was evaluated using training-derived
event thresholds.

### RELIANCE

- High-news-event OOS coverage: 10.00%
- Actual directional prediction coverage: 0.81%
- Balanced accuracy: 0.3293
- Macro-F1: 0.2654

### TCS

- High-news-event OOS coverage: 2.38%
- Actual directional prediction coverage: 0.00%
- Balanced accuracy: 0.3333
- Macro-F1: 0.1916

### Decision

The selective policy was **not promoted**.

The policy became too sparse to provide useful directional coverage.

---

# Overall Phase 2 News Finding

The evidence does **not** support the statement that news sentiment
generally improves market prediction.

The defensible conclusion is:

> News sentiment provides preliminary, asset-specific incremental information
> for RELIANCE, particularly within a trailing one-hour window, but comparable
> incremental benefit is not consistently observed for TCS. News intensity and
> selective event-regime variants do not provide sufficient robust evidence for
> promotion into the general predictive feature set.

---

## Final News Decision

### RELIANCE

**Status: Exploratory candidate**

The matched-sample experiment provides evidence of incremental information:

- Balanced accuracy: +0.0552
- Macro-F1: +0.0887

However, the OOS sample contains only 135 news-supported observations.

Therefore this should remain a research finding rather than a confirmed
generalizable production signal.

### TCS

**Status: Not supported**

The matched-sample experiment does not show consistent improvement, and the
event-intensity experiment is negative.

No incremental news benefit should be claimed for TCS.

---

## Features Not Promoted

The following were investigated but should not be added to the frozen
market feature set:

- News intensity as a universal additive block
- News burst as a general predictive feature
- High-news event regime
- Selective news directional policy

---

## Holdout Protection

The locked observations beginning 2026-08-10 remain protected.

They were not used for:

- feature selection
- threshold selection
- model fitting
- event-regime selection
- OOS metric comparison

The locked set should remain untouched until the research design explicitly
calls for final evaluation.

---

## Phase 2 News Status

**CLOSED — EXPLORATORY NEWS INVESTIGATION**

The current evidence is sufficient to stop iterating on the same news feature
family without introducing a new research hypothesis or information source.

Recommended next research direction: evaluate a genuinely different
information modality, such as social-media sentiment, under the same leakage
controls and chronological evaluation framework.
"""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        output,
        encoding="utf-8",
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()