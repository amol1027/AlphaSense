# Starting AlphaSense

This guide starts the local dashboard and the market-data shadow runner on Windows. The dashboard serves predictions and can fetch Upstox candles on demand. The optional shadow runner refreshes candles automatically every 15 minutes and records prospective predictions.

## 1. Open two PowerShell terminals

In both terminals, go to the project folder:

```powershell
cd "D:\projects\Research Project\AlphaSense"
```

If dependencies have not been installed in the project virtual environment yet, do this once:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Check Upstox access

The project root `.env` file must contain a valid access token:

```text
UPSTOX_ACCESS_TOKEN=your_token
```

Keep the token private. Check it before starting the runner:

```powershell
.\.venv\Scripts\python.exe scripts/check_upstox_token.py
```

Upstox access tokens expire daily. If the check reports an expired or missing token, renew it through the project's existing OAuth process and update `.env`. The shadow runner enforces the same gate at boot and refuses to start on a bad token, so always renew before restarting it.

## 3. Start the shadow runner

In the first terminal:

```powershell
.\.venv\Scripts\python.exe scripts/run_shadow.py
```

The runner refreshes completed 15-minute candles about every 15 minutes, including a final cycle through 3:45 pm IST to capture the 3:15–3:30 pm candle. It labels outcomes once a full future hour is available and logs fresh predictions. The dashboard uses Upstox's market stream for live chart updates; its historical intraday candle endpoint supplies chart history. The local prediction cache retains a rolling seven days; the original research data remains unchanged. Leave this terminal running. Press **Ctrl+C** to stop it.

The first refresh seeds the cache from the existing market files, then adds current Upstox candles. Predictions are logged only when the latest bar is no more than 45 minutes old. If the market is closed or no completed bars are available, the runner waits for its next cycle.

## 4. Start the prediction API

In the second terminal:

```powershell
.\.venv\Scripts\python.exe -m src.prediction.server
```

This API supplies forecasts and market data to the React website; it is no longer the dashboard page itself. To run the website and open its Markets page, follow the three-terminal steps in [`website/README.md`](../website/README.md). Keep the shadow runner and prediction API running while using the market dashboard. The separate account API is only needed for registration and sign-in. Stop each service with **Ctrl+C** in its terminal.

## Health checks

Both APIs expose readiness endpoints for supervisors and uptime monitors:

```powershell
(Invoke-RestMethod http://127.0.0.1:8000/health).status      # ok | degraded
(Invoke-RestMethod http://127.0.0.1:8010/api/auth/health -SkipHttpErrorCheck).StatusCode  # 200 | 503
```

The prediction `/health` returns HTTP 503 with per-check detail (`market_cache`, `model_artifacts`, `upstox_token_configured`, `webapp`) instead of an unconditional 200. On Linux, `deploy/systemd/` holds units for all three services (prediction, auth, shadow) with restart policies; adjust `User=`/paths before installing.

## Common startup issues

- **“Live market cache is missing”**: start the shadow runner and wait for a successful refresh.
- **“UPSTOX_ACCESS_TOKEN is not set” or expired**: update the root `.env` through the normal Upstox OAuth process, then rerun the token check.
- **No predictions yet**: wait for completed bars; the model needs enough history to build its features, and prediction logging skips stale bars.
- **Missing model artifact**: from the project root, run `.\.venv\Scripts\python.exe scripts/train_volatility_model.py`, then restart the dashboard. This regenerates the local frozen-model bundles from pre-locked data.
- **Port 8000 is already in use**: stop the other local prediction service using that port, then restart the API.
