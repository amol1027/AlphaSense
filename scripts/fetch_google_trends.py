"""Fetch Google Trends daily search interest per asset (geo=IN).

One keyword per asset (single-keyword queries scale 0-100 independently;
all uses are within-asset, so comparability is preserved). Yearly chunks
to stay well under response limits; sleeps between requests.
Output: data/raw/attention/google_trends_daily.csv (asset,date,interest).
"""

import time
from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq

KEYWORDS = {
    "RELIANCE": "Reliance Industries",
    "TCS": "Tata Consultancy Services",
    "HDFCBANK": "HDFC Bank",
    "INFY": "Infosys",
    "ICICIBANK": "ICICI Bank",
}

OUTPUT = Path("data/raw/attention/google_trends_daily.csv")
# Six-month chunks: multi-year queries degrade to weekly resolution,
# so chunking forces daily rows.
CHUNKS = [
    ("2024-01-01", "2024-06-30"),
    ("2024-07-01", "2024-12-31"),
    ("2025-01-01", "2025-06-30"),
    ("2025-07-01", "2025-12-31"),
    ("2026-01-01", "2026-06-30"),
    ("2026-07-01", "2026-09-27"),
]


def fetch_keyword(pytrends: TrendReq, keyword: str) -> pd.DataFrame:
    frames = []
    for start, end in CHUNKS:
        timeframe = f"{start} {end}"
        for attempt in range(4):
            try:
                pytrends.build_payload([keyword], timeframe=timeframe, geo="IN")
                df = pytrends.interest_over_time()
                break
            except Exception as exc:
                print(f"  {timeframe} attempt {attempt}: {type(exc).__name__}; sleeping")
                time.sleep(30 * (attempt + 1))
        else:
            raise RuntimeError(f"Failed {keyword} {timeframe}")
        if "isPartial" in df.columns:
            df = df.drop(columns=["isPartial"])
        frames.append(df.rename(columns={keyword: "interest"}))
        time.sleep(10)
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    out.index = pd.to_datetime(out.index, utc=True)
    return out


def main() -> None:
    pytrends = TrendReq(hl="en-US", tz=330, retries=2, backoff_factor=1.0)
    all_rows = []
    for asset, keyword in KEYWORDS.items():
        print(f"{asset} ({keyword})...", flush=True)
        df = fetch_keyword(pytrends, keyword)
        df["asset"] = asset
        df["date"] = df.index.date
        all_rows.append(df[["asset", "date", "interest"]].reset_index(drop=True))
        print(f"  {len(df)} days, {df.index.min().date()} -> {df.index.max().date()}", flush=True)
        time.sleep(15)
    result = pd.concat(all_rows, ignore_index=True).sort_values(["asset", "date"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(f"Saved {len(result)} rows -> {OUTPUT}")
    print(result.groupby("asset")["interest"].agg(["count", "mean"]).to_string())


if __name__ == "__main__":
    main()
