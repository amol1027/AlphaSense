"""Contract tests for the read-only volatility prediction service."""

import pandas as pd
import pytest

from src.features.market_features import add_normalized_market_features

from scripts.evaluate_phase2_news_event_regimes import load_market
from scripts.validate_phase4_volatility_pilot import add_trailing_vol
from src.prediction.artifacts import FEATURES, load_bundle
from src.prediction.features import MIN_BARS, build_serving_features
from src.prediction.service import predict_range_regime


def _bars(asset="TCS", n=30):
    df = load_market()
    sub = df[df["asset"] == asset].sort_values("timestamp").tail(n).reset_index(drop=True)
    sub["timestamp"] = pd.to_datetime(sub["timestamp"], utc=True)
    return sub


def test_serving_features_match_validation_builders():
    bars = _bars(n=40)
    full = add_trailing_vol(add_normalized_market_features(bars.copy()))
    serving = build_serving_features(bars.copy())
    row_full = full.iloc[-1]
    row_serv = serving.iloc[-1]
    for col in FEATURES:
        assert row_serv[col] == pytest.approx(row_full[col], rel=1e-9, abs=1e-12)


def test_features_use_only_history():
    """Row i features from the full frame must equal those from frame truncated at i."""
    bars = _bars(n=40)
    full_feats = build_serving_features(bars.copy())
    trunc_feats = build_serving_features(bars.head(30).copy().reset_index(drop=True))
    for col in FEATURES:
        assert trunc_feats[col].iloc[-1] == pytest.approx(
            full_feats[col].iloc[29], rel=1e-9, abs=1e-12
        )


def test_prediction_is_deterministic():
    bars = _bars()
    assert predict_range_regime("TCS", bars) == predict_range_regime("TCS", bars.copy())


def test_unknown_asset_rejected():
    bars = _bars()
    bars["asset"] = "UNKNOWN_ASSET"
    with pytest.raises(ValueError, match="[Nn]o trained artifact"):
        predict_range_regime("UNKNOWN_ASSET", bars)


def test_insufficient_bars_rejected():
    with pytest.raises(ValueError, match="at least"):
        predict_range_regime("TCS", _bars(n=MIN_BARS - 1))


def test_mixed_assets_rejected():
    bars = _bars(n=30)
    other = _bars(asset="INFY", n=5)
    mixed = pd.concat([bars, other], ignore_index=True)
    with pytest.raises(ValueError, match="single asset"):
        predict_range_regime("TCS", mixed)


def test_response_contract():
    res = predict_range_regime("TCS", _bars())
    assert set(res) == {
        "asset", "prediction_timestamp", "target", "probability_high_range",
        "prediction", "latest_price_change_pct", "range_median", "n_train",
        "data_age_days", "stale", "data_used",
    }
    assert res["asset"] == "TCS"
    assert res["prediction"] in (0, 1)
    assert 0.0 <= res["probability_high_range"] <= 1.0
    assert isinstance(res["stale"], bool)
    assert res["data_age_days"] >= 0.0
    bundle = load_bundle("TCS")
    assert res["range_median"] == pytest.approx(bundle.range_median)
    data_used = res["data_used"]
    assert data_used["feature_names"] == list(FEATURES)
    assert set(data_used["features"]) == set(FEATURES)
    assert data_used["bar_count"] == 30
    assert data_used["decision_threshold"] == 0.5


def test_stale_flag_on_old_bars():
    bars = _bars()
    bars["timestamp"] = pd.Timestamp("2020-01-01", tz="UTC") + (
        bars["timestamp"] - bars["timestamp"].iloc[0]
    )
    res = predict_range_regime("TCS", bars)
    assert res["stale"] is True
    assert res["data_age_days"] > 7.0


def test_no_legacy_dashboard_html():
    """The legacy inline dashboard is gone; React (website/) is the only UI."""
    import src.prediction.server as server

    assert not hasattr(server, "DASHBOARD_HTML")
    assert (server.PROJECT_ROOT / "website" / "src" / "MarketDashboard.tsx").is_file()


def test_webapp_path_resolution():
    from src.prediction.server import WEBAPP_DIR, _webapp_path

    assert _webapp_path("/../secret.txt") is None
    assert _webapp_path("/") is None
    assert _webapp_path("/does-not-exist-12345.js") is None
    # Existing build output resolves when dist/ is present.
    if (WEBAPP_DIR / "index.html").is_file():
        resolved = _webapp_path("/index.html")
        assert resolved is not None and resolved.name == "index.html"
