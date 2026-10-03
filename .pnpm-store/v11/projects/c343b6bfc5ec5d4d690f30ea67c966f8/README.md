# AlphaSense research site

This is a Vite + React + TypeScript site for the AlphaSense research project. The research note and interactive stock dashboard share one React site. Prediction and market-data logic stay in the Python service, while account records and revocable sessions use MySQL through the separate FastAPI service under `src/auth`.

## Run locally on Windows

The React site uses the existing Python prediction service for market data and forecasts, plus a separate account API for registration and sign-in. Start the prediction service and website in separate terminals; the account API is also needed for account features. MySQL must be running, and the account tables must already be initialized. Follow [`docs/ACCOUNT_AUTH.md`](../docs/ACCOUNT_AUTH.md) once to configure MySQL, install the Python dependencies, and create the tables.

### 1. Start the prediction service

Open PowerShell in the repository root (`D:\projects\Research Project\AlphaSense`), activate the project's Python environment, then run:

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.prediction.server
```

The prediction service listens on port `8000`. The `/markets` page uses it for forecasts, stock charts, history, live feed, and manual data refresh. If you want automatic candle refreshes and shadow prediction logging, start `python scripts/run_shadow.py` in another terminal as well. That optional runner needs a valid `UPSTOX_ACCESS_TOKEN` in the root `.env`.

### 2. Start the account API

In another PowerShell terminal at the repository root, activate the same Python environment if needed and start the account service:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn src.auth.app:app --host 127.0.0.1 --port 8010
```

Keep this terminal open to use registration and sign-in. If port `8010` is already in use, stop the other account API process with **Ctrl+C** before starting it again.

### 3. Start the React website

Open a third PowerShell terminal in the repository's `website` folder:

```powershell
cd "D:\projects\Research Project\AlphaSense\website"
pnpm install
pnpm dev -- --host 127.0.0.1
```

Open the Vite address printed in the terminal, normally [http://127.0.0.1:5173](http://127.0.0.1:5173), then choose **Markets** in the site navigation. Leave all three terminals running. Stop a service with **Ctrl+C** in its terminal.

The site proxies market API requests to `http://127.0.0.1:8000` and account API requests to `http://127.0.0.1:8010` through same-origin `/market-api` and `/auth-api` paths (see `vite.config.ts` and `.env.example`); keep both Python services running. To point at another account API URL directly, set `VITE_AUTH_API_BASE_URL` in `website/.env.local` and add the frontend's exact origin to `AUTH_ALLOWED_ORIGINS` in the repository root `.env`, then restart the account API.

### Production preview

Create and serve a local production build from the `website` folder. Keep both Python APIs running while using it:

```powershell
pnpm build
pnpm preview -- --host 127.0.0.1
```

Vite normally serves the preview at [http://127.0.0.1:4173](http://127.0.0.1:4173). The generated `dist/` directory can be deployed to Vercel, Netlify, or GitHub Pages with an SPA fallback for `/login`, `/register`, and `/markets`. Configure the deployed host to proxy `/market-api/*` to the Python prediction service and `/auth-api/*` to the account API; the local Vite proxy is only active in development and preview. The Python prediction service also hosts `website/dist/` itself (same origin, no proxy needed) once you run `pnpm build`.

## Current slice

The site contains the public research note, the live stock dashboard at `/markets`, and registration/sign-in pages. The rest of the research narrative is still staged for review.

## Design system

- Palette, typography, and motion tokens: `src/tokens.css`
- Page components and current content: `src/main.tsx`
- Layout, responsive behavior, and interaction states: `src/styles.css`
- React market dashboard and stock details: `src/MarketDashboard.tsx`, `src/market-dashboard.css`
- Authentication requests and user session types: `src/auth-client.ts`
- Registration and sign-in screens: `src/AuthPage.tsx`
- MySQL-backed authentication API and schema: `../src/auth/`

Research claims in this slice are drawn from the root project README and `docs/phase_4_pilot.md`. Candle marks are illustrative; no prices or accuracy metrics are implied.

See [`docs/ACCOUNT_AUTH.md`](../docs/ACCOUNT_AUTH.md) for detailed MySQL setup and production configuration.
