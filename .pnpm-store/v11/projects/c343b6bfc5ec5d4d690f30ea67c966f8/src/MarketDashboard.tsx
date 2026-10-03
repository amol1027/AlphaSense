import { Fragment, useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { Activity, ArrowDownRight, ArrowUpRight, RefreshCw, Star, X } from 'lucide-react'
import './market-dashboard.css'

const ASSETS = [
  { symbol: 'RELIANCE', name: 'Reliance Industries', short: 'Reliance' },
  { symbol: 'TCS', name: 'Tata Consultancy Services', short: 'TCS' },
  { symbol: 'HDFCBANK', name: 'HDFC Bank', short: 'HDFC Bank' },
  { symbol: 'INFY', name: 'Infosys', short: 'Infosys' },
  { symbol: 'ICICIBANK', name: 'ICICI Bank', short: 'ICICI Bank' },
] as const

type Asset = typeof ASSETS[number]['symbol']
type Interval = '1m' | '5m' | '15m' | '1h'

// Model inputs grouped for the inspection table. Mirrors the frozen 9-feature
// volatility set; names missing from a response are skipped when rendering.
const FEATURE_GROUPS: Array<{ title: string; names: string[] }> = [
  { title: 'Returns', names: ['return_15m', 'return_30m', 'return_1h', 'close_open_return'] },
  { title: 'Range', names: ['high_low_range', 'range_mean_1h', 'range_max_1h'] },
  { title: 'Volume & volatility', names: ['volume_change', 'ret_std_1h'] },
]
type Bar = { time: number; open: number; high: number; low: number; close: number }
type Prediction = {
  asset: Asset
  prediction_timestamp: string
  target: string
  probability_high_range: number
  prediction: 0 | 1
  latest_price_change_pct: number
  range_median: number
  n_train: number
  data_age_days: number
  stale: boolean
  data_used?: { features?: Record<string, number>; feature_names?: string[]; bar_count?: number; trained_at?: string }
  error?: string
}
type Feed = { state: 'fresh' | 'delayed' | 'stale' | 'waiting'; latest_bar_timestamp: string | null; age_minutes: number | null; last_refresh_at: string | null }
type HistoryItem = { asset: Asset; prediction_timestamp: string; probability: number; prediction: 0 | 1; correct: boolean | null; outcome_status: string }
type Snapshot = { feed: Feed; predictions: Partial<Record<Asset, Prediction>>; history: HistoryItem[]; checked_at: string }
type ChartData = { asset: Asset; interval: Interval; bars: Bar[]; minute_bars: Bar[]; message?: string }
type NewsItem = { headline: string; source: string; published_at: string; url: string; text?: string }
type NewsResponse = { asset: Asset; items: NewsItem[]; total: number; limit: number; offset: number; status: { last_refresh_at: string | null; cache_rows?: number; provider_errors?: Record<string, string> }; message?: string }

const NEWS_PAGE_SIZE = 10

const API = (import.meta.env.VITE_MARKET_API_BASE_URL || '/market-api').replace(/\/$/, '')
const DEFAULT_FEED: Feed = { state: 'waiting', latest_bar_timestamp: null, age_minutes: null, last_refresh_at: null }

function ist(iso?: string | null, includeDate = false) {
  if (!iso) return '—'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return '—'
  return date.toLocaleString('en-IN', {
    timeZone: 'Asia/Kolkata',
    ...(includeDate ? { day: 'numeric' as const, month: 'short' as const } : {}),
    hour: '2-digit', minute: '2-digit', hourCycle: 'h23',
  })
}

function price(value: number) {
  return `₹${value.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
}

function formatAge(ageMinutes: number | null | undefined) {
  if (ageMinutes == null) return ''
  if (ageMinutes < 60) return `${Math.round(ageMinutes)} min ago`
  if (ageMinutes < 48 * 60) {
    const hours = ageMinutes / 60
    return `${hours >= 10 ? Math.round(hours) : hours.toFixed(1)} hours ago`
  }
  return `${(ageMinutes / 1440).toFixed(1)} days ago`
}

function predictedOutcome(prediction: Prediction) {
  return prediction.prediction === 1 ? prediction.probability_high_range : 1 - prediction.probability_high_range
}

type MarketView = { stock: Asset | null; interval: Interval; filter: string; sort: string }

function parseMarketSearch(search: string): MarketView {
  const params = new URLSearchParams(search)
  const rawStock = params.get('stock')
  const stock = ASSETS.some(item => item.symbol === rawStock) ? (rawStock as Asset) : null
  const rawInterval = params.get('interval')
  const interval: Interval = rawInterval === '1m' || rawInterval === '5m' || rawInterval === '1h' ? rawInterval : '15m'
  const rawFilter = params.get('show')
  const filter = rawFilter && (rawFilter === 'all' || ASSETS.some(item => item.symbol === rawFilter)) ? rawFilter : 'all'
  const rawSort = params.get('order')
  const sort = rawSort === 'high' || rawSort === 'low' ? rawSort : 'default'
  return { stock, interval, filter, sort }
}

function marketUrl(view: MarketView) {
  const params = new URLSearchParams()
  if (view.stock) params.set('stock', view.stock)
  if (view.interval !== '15m') params.set('interval', view.interval)
  if (view.filter !== 'all') params.set('show', view.filter)
  if (view.sort !== 'default') params.set('order', view.sort)
  const query = params.toString()
  return `/markets${query ? `?${query}` : ''}`
}

function commitMarketUrl(view: MarketView, replace: boolean) {
  const url = marketUrl(view)
  if (window.location.pathname + window.location.search !== url) {
    if (replace) window.history.replaceState({}, '', url)
    else window.history.pushState({}, '', url)
  }
}

function updateBarList(bars: Bar[], incoming: Bar) {
  const next = bars.filter(item => item.time !== incoming.time)
  next.push(incoming)
  next.sort((a, b) => a.time - b.time)
  return next.slice(-600)
}

function bucketStart(time: number, minutes: number) {
  // 03:45 UTC == 09:15 IST session open. Buckets anchor there so 60m buckets
  // land on 09:15/10:15/... matching the prediction windows. UTC date equals
  // the IST date during session hours (03:45-10:00 UTC), so this is safe.
  const tick = new Date(time * 1000)
  const sessionAnchor = Date.UTC(tick.getUTCFullYear(), tick.getUTCMonth(), tick.getUTCDate(), 3, 45) / 1000
  return sessionAnchor + Math.floor((time - sessionAnchor) / (minutes * 60)) * minutes * 60
}

function SnapshotChart({ bars, style }: { bars: Bar[]; style: 'candles' | 'line' }) {
  if (!bars.length) return <div className="market-chart-empty">No chart bars are available for this interval.</div>
  const width = 1000, height = 300, left = 66, right = 18, top = 15, bottom = 27
  const values = bars.flatMap(bar => [bar.low, bar.high])
  const rawMin = Math.min(...values), rawMax = Math.max(...values)
  const span = rawMax - rawMin || Math.max(rawMax * .005, 1)
  const min = rawMin - span * .08, max = rawMax + span * .08
  const plotWidth = width - left - right, plotHeight = height - top - bottom
  const x = (index: number) => left + (bars.length < 2 ? plotWidth / 2 : index / (bars.length - 1) * plotWidth)
  const y = (value: number) => top + (max - value) / (max - min) * plotHeight
  const bodyWidth = Math.max(1.2, Math.min(7, plotWidth / bars.length * .58))
  const linePoints = bars.map((bar, index) => `${x(index)},${y(bar.close)}`).join(' ')

  return <div className="market-chart-scroll">
    <svg className="market-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${style === 'line' ? 'Line' : 'Candlestick'} price chart with ${bars.length} bars`}>
      {[0, 1, 2, 3].map(step => {
        const value = max - step / 3 * (max - min), lineY = top + step / 3 * plotHeight
        return <g key={step}><line x1={left} x2={width - right} y1={lineY} y2={lineY} className="market-gridline" /><text x={left - 9} y={lineY + 4} textAnchor="end" className="market-axis-label">{price(value)}</text></g>
      })}
      {style === 'line' ? <polyline points={linePoints} fill="none" className="market-line" /> : bars.map((bar, index) => {
        const positive = bar.close >= bar.open, colorClass = positive ? 'up' : 'down'
        const bodyY = Math.min(y(bar.open), y(bar.close)), bodyHeight = Math.max(1.4, Math.abs(y(bar.open) - y(bar.close)))
        return <g key={bar.time} className={`market-candle ${colorClass}`}>
          <title>{`${ist(new Date(bar.time * 1000).toISOString(), true)} IST · O ${price(bar.open)} · H ${price(bar.high)} · L ${price(bar.low)} · C ${price(bar.close)}`}</title>
          <line x1={x(index)} x2={x(index)} y1={y(bar.high)} y2={y(bar.low)} className="market-wick" />
          <rect x={x(index) - bodyWidth / 2} y={bodyY} width={bodyWidth} height={bodyHeight} className="market-candle-body" />
        </g>
      })}
      {[0, Math.floor((bars.length - 1) / 2), bars.length - 1].filter((value, pos, arr) => arr.indexOf(value) === pos).map(index => {
        const bar = bars[index]
        return <text key={bar.time} x={x(index)} y={height - 6} textAnchor={index === 0 ? 'start' : index === bars.length - 1 ? 'end' : 'middle'} className="market-axis-label">{ist(new Date(bar.time * 1000).toISOString()).slice(0, 5)}</text>
      })}
    </svg>
  </div>
}

export function MarketDashboard({ onNavigate, search }: { onNavigate: (path: string) => void; search: string }) {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [serviceError, setServiceError] = useState('')
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState(() => parseMarketSearch(window.location.search).filter)
  const [sort, setSort] = useState(() => parseMarketSearch(window.location.search).sort)
  const [refreshing, setRefreshing] = useState(false)
  const [refreshMessage, setRefreshMessage] = useState('')
  const [detailAsset, setDetailAsset] = useState<Asset | null>(() => parseMarketSearch(window.location.search).stock)
  const [interval, setInterval] = useState<Interval>(() => parseMarketSearch(window.location.search).interval)
  const [chartStyle, setChartStyle] = useState<'candles' | 'line'>('candles')
  const [chart, setChart] = useState<ChartData | null>(null)
  const [chartLoading, setChartLoading] = useState(false)
  const [liveStatus, setLiveStatus] = useState('Live feed connects when a stock is open')
  const [news, setNews] = useState<NewsItem[]>([])
  const [newsLoading, setNewsLoading] = useState(false)
  const [newsRefreshAt, setNewsRefreshAt] = useState<string | null>(null)
  const [newsPage, setNewsPage] = useState(0)
  const [newsTotal, setNewsTotal] = useState(0)
  const [starredOnly, setStarredOnly] = useState(false)
  const [starred, setStarred] = useState<Asset[]>(() => {
    try {
      const saved = JSON.parse(window.localStorage.getItem('alphasense-watchlist') ?? '[]') as string[]
      return ASSETS.map(item => item.symbol).filter(symbol => saved.includes(symbol))
    } catch {
      return []
    }
  })

  function toggleStar(asset: Asset) {
    setStarred(current => {
      const next = current.includes(asset) ? current.filter(symbol => symbol !== asset) : [...current, asset]
      try {
        window.localStorage.setItem('alphasense-watchlist', JSON.stringify(next))
      } catch {
        /* private mode: watchlist stays in memory */
      }
      return next
    })
  }
  const triggerRef = useRef<HTMLElement | null>(null)
  const dialogRef = useRef<HTMLElement | null>(null)

  const openDetail = useCallback((asset: Asset, trigger: HTMLElement | null) => {
    // Failed cards are inert: no dialog, no chart fetch.
    const prediction = snapshot?.predictions[asset]
    if (!prediction || prediction.error) return
    triggerRef.current = trigger
    const current = parseMarketSearch(window.location.search)
    setInterval('15m')
    setDetailAsset(asset)
    // Push so the browser back button closes the dialog.
    commitMarketUrl({ ...current, interval: '15m', stock: asset }, false)
  }, [snapshot])

  const closeDetail = useCallback(() => {
    setDetailAsset(null)
    // Replace: the ?stock entry is consumed, so back skips it.
    const current = parseMarketSearch(window.location.search)
    commitMarketUrl({ ...current, stock: null }, true)
  }, [])

  // View controls write through to the URL (replace, no history spam) so a
  // copied link reproduces the same filter/sort/interval/open stock.
  function updateView(patch: { filter?: string; sort?: string; interval?: Interval }) {
    const nextFilter = patch.filter ?? filter
    const nextSort = patch.sort ?? sort
    const nextInterval = patch.interval ?? interval
    setFilter(nextFilter)
    setSort(nextSort)
    setInterval(nextInterval)
    commitMarketUrl({ stock: detailAsset, filter: nextFilter, sort: nextSort, interval: nextInterval }, true)
  }

  // Follow the URL when it changes outside this view (back/forward buttons,
  // header navigation). Own writers above bypass App state, so this loop is
  // one-way: URL -> state.
  useEffect(() => {
    const view = parseMarketSearch(search)
    setFilter(view.filter)
    setSort(view.sort)
    setInterval(view.interval)
    setDetailAsset(view.stock)
  }, [search])

  const loadSnapshot = useCallback(async (quiet = false) => {
    if (!quiet) setLoading(true)
    try {
      const response = await fetch(`${API}/dashboard`)
      if (!response.ok) throw new Error(`Prediction service returned ${response.status}.`)
      const data = await response.json() as Snapshot
      setSnapshot(data)
      setServiceError('')
    } catch {
      setServiceError('Can’t reach the market service on port 8000. Start the Python prediction API and try again.')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadSnapshot()
    const timer = window.setInterval(() => { void loadSnapshot(true) }, 60_000)
    return () => window.clearInterval(timer)
  }, [loadSnapshot])

  useEffect(() => {
    if (!detailAsset) return
    const controller = new AbortController()
    setChartLoading(true)
    setChart(null)
    fetch(`${API}/chart?asset=${encodeURIComponent(detailAsset)}&interval=${interval}`, { signal: controller.signal })
      .then(async response => {
        const data = await response.json() as ChartData
        if (!response.ok) throw new Error((data as ChartData & { error?: string }).error || 'Price history could not be loaded.')
        setChart(data)
      })
      .catch(error => {
        if (error.name !== 'AbortError') setChart({ asset: detailAsset, interval, bars: [], minute_bars: [], message: error.message })
      })
      .finally(() => { if (!controller.signal.aborted) setChartLoading(false) })
    return () => controller.abort()
  }, [detailAsset, interval])

  useEffect(() => {
    if (!detailAsset) return
    const controller = new AbortController()
    setNewsLoading(true)
    fetch(`${API}/news?asset=${encodeURIComponent(detailAsset)}&limit=${NEWS_PAGE_SIZE}&offset=${newsPage * NEWS_PAGE_SIZE}`, { signal: controller.signal })
      .then(async response => {
        const data = await response.json() as NewsResponse
        if (!response.ok) throw new Error('News could not be loaded.')
        setNews(data.items ?? [])
        setNewsTotal(typeof data.total === 'number' ? data.total : (data.items ?? []).length)
        setNewsRefreshAt(data.status?.last_refresh_at ?? null)
      })
      .catch(error => {
        if (error.name !== 'AbortError') {
          setNews([])
          setNewsTotal(0)
          setNewsRefreshAt(null)
        }
      })
      .finally(() => { if (!controller.signal.aborted) setNewsLoading(false) })
    return () => controller.abort()
  }, [detailAsset, newsPage])

  // A new stock starts back on the first headlines page.
  useEffect(() => { setNewsPage(0) }, [detailAsset])

  useEffect(() => {
    if (!detailAsset) return
    const stream = new EventSource(`${API}/stream`)
    stream.onmessage = event => {
      try {
        const message = JSON.parse(event.data) as { type: string; status?: string; asset?: Asset; bar?: Bar }
        if (message.type === 'status' && message.status) setLiveStatus(`Live feed ${message.status}`)
        if (message.type !== 'candle' || message.asset !== detailAsset || !message.bar) return
        setLiveStatus('Live price updates received')
        setChart(current => {
          if (!current || current.asset !== detailAsset) return current
          const minuteBars = updateBarList(current.minute_bars, message.bar!)
          if (current.interval === '1m') return { ...current, minute_bars: minuteBars, bars: updateBarList(current.bars, message.bar!) }
          const minutes = current.interval === '5m' ? 5 : current.interval === '15m' ? 15 : 60
          const start = bucketStart(message.bar!.time, minutes)
          const grouped = minuteBars.filter(bar => bar.time >= start && bar.time < start + minutes * 60)
          // Only publish a bucket once all of its 1-minute bars are present,
          // mirroring the server's strict count == minutes rule. Partial
          // buckets stay in minuteBars so the chart never shows a false
          // "complete" candle built from a single live tick.
          if (grouped.length < minutes) return { ...current, minute_bars: minuteBars }
          const aggregate: Bar = { time: start, open: grouped[0].open, high: Math.max(...grouped.map(bar => bar.high)), low: Math.min(...grouped.map(bar => bar.low)), close: grouped[grouped.length - 1].close }
          return { ...current, minute_bars: minuteBars, bars: updateBarList(current.bars, aggregate) }
        })
      } catch { /* ignore malformed stream messages */ }
    }
    stream.onerror = () => setLiveStatus('Live feed is reconnecting; cached chart data is still available')
    return () => stream.close()
  }, [detailAsset])

  useEffect(() => {
    if (!detailAsset) return
    // Initial focus goes to the close button so keyboard users start inside.
    dialogRef.current?.querySelector<HTMLElement>('.market-dialog-close')?.focus()
    const trapFocus = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        closeDetail()
        return
      }
      if (event.key !== 'Tab' || !dialogRef.current) return
      const focusables = Array.from(
        dialogRef.current.querySelectorAll<HTMLElement>(
          'a[href], button:not(:disabled), input, select, textarea, summary, [tabindex]:not([tabindex="-1"])',
        ),
      ).filter(element => element.getClientRects().length > 0)
      if (!focusables.length) return
      const first = focusables[0], last = focusables[focusables.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }
    window.addEventListener('keydown', trapFocus)
    return () => window.removeEventListener('keydown', trapFocus)
  }, [detailAsset, closeDetail])

  // Return focus to the card that opened the dialog.
  useEffect(() => {
    if (detailAsset) return
    const trigger = triggerRef.current
    triggerRef.current = null
    if (trigger && document.contains(trigger)) trigger.focus()
  }, [detailAsset])

  const visibleAssets = useMemo(() => {
    let items = ASSETS.filter(item => (filter === 'all' || item.symbol === filter) && (!starredOnly || starred.includes(item.symbol)))
    if (sort !== 'default') items = [...items].sort((a, b) => {
      const pa = snapshot?.predictions[a.symbol], pb = snapshot?.predictions[b.symbol]
      // Sort by P(larger moves), not by confidence in the predicted class,
      // so "higher" really means a higher larger-move chance.
      const probA = pa && !pa.error ? pa.probability_high_range : -1
      const probB = pb && !pb.error ? pb.probability_high_range : -1
      return sort === 'high' ? probB - probA : probA - probB
    })
    return items
  }, [filter, sort, snapshot, starredOnly, starred])

  async function refreshMarket() {
    setRefreshing(true)
    setRefreshMessage('Fetching latest market candles from Upstox…')
    try {
      const response = await fetch(`${API}/refresh`, { method: 'POST' })
      const result = await response.json() as { error?: string; assets_updated?: number; predictions_added?: number; outcomes_scored?: number; shadow_error?: string; news?: { cache_rows?: number; fresh_fetched?: number }; news_error?: string }
      if (!response.ok) throw new Error(result.error || `Refresh failed (${response.status}).`)
      await loadSnapshot(true)
      if (detailAsset) {
        try {
          const newsResponse = await fetch(`${API}/news?asset=${encodeURIComponent(detailAsset)}&limit=${NEWS_PAGE_SIZE}&offset=0`)
          const newsData = await newsResponse.json() as NewsResponse
          if (newsResponse.ok) {
            setNewsPage(0)
            setNews(newsData.items ?? [])
            setNewsTotal(typeof newsData.total === 'number' ? newsData.total : (newsData.items ?? []).length)
            setNewsRefreshAt(newsData.status?.last_refresh_at ?? null)
          }
        } catch { /* news refresh is best-effort */ }
      }
      setRefreshMessage(`Updated ${result.assets_updated ?? 0} stocks · ${result.predictions_added ?? 0} new logged forecasts · ${result.outcomes_scored ?? 0} outcomes scored · ${result.news?.fresh_fetched ?? 0} fresh headlines${result.shadow_error ? ` · shadow log: ${result.shadow_error}` : ''}${result.news_error ? ` · news: ${result.news_error}` : ''}`)
    } catch (error) {
      setRefreshMessage(error instanceof Error ? error.message : 'Market refresh failed.')
    } finally {
      setRefreshing(false)
    }
  }

  const newsPageCount = Math.max(1, Math.ceil(newsTotal / NEWS_PAGE_SIZE))
  const feed = snapshot?.feed ?? DEFAULT_FEED
  const history = snapshot?.history ?? []
  const latest = snapshot?.checked_at ? ist(snapshot.checked_at) : '—'
  const assetName = (asset: string) => ASSETS.find(item => item.symbol === asset)?.name ?? asset
  const selectedPrediction = detailAsset ? snapshot?.predictions[detailAsset] : undefined
  const change = selectedPrediction?.latest_price_change_pct ?? 0

  return <main id="market-main" tabIndex={-1} className="market-page">
    <section className="market-masthead">
      <div className="market-kicker"><span>RESEARCH TERMINAL / 01</span><span>NSE · 15 MIN BARS · IST</span></div>
      <div className="market-title-row"><div><p className="eyebrow">LIVE MARKET / PILOT MODEL</p><h1>Range, not direction.</h1></div><p className="market-lede">Five Indian equities. One question: is the coming hour likely to be quieter or more active than usual?</p></div>
      <p className="market-honesty"><span>†</span> Experimental research only. Probabilities describe volatility regime, never whether a share price will rise or fall.</p>
    </section>

    <section className={`market-feed ${feed.state}`} aria-live="polite">
      <span className="market-feed-indicator" />
      <div className="market-feed-copy"><strong>{feed.state === 'fresh' ? 'Market data is fresh' : feed.state === 'delayed' ? 'Market data is delayed' : feed.state === 'stale' ? 'Market data is stale' : 'Waiting for market data'}</strong><span>{feed.latest_bar_timestamp ? `Last completed bar ${ist(feed.latest_bar_timestamp, true)} IST${feed.age_minutes == null ? '' : ` · ${formatAge(feed.age_minutes)}`}` : 'Start the shadow runner or fetch a one-time market refresh.'}</span></div>
      <div className="market-feed-meta"><span>LAST REFRESH</span><strong>{ist(feed.last_refresh_at, true)}</strong></div>
    </section>

    <section className="market-outlook" aria-labelledby="outlook-title">
      <div className="market-section-head"><div><p className="eyebrow">NEXT HOUR / MODEL ESTIMATE</p><h2 id="outlook-title">Latest outlook</h2></div><div className="market-section-meta"><span>Checked {latest} IST</span><button type="button" className="market-refresh" onClick={() => void refreshMarket()} disabled={refreshing}><RefreshCw size={14} className={refreshing ? 'is-spinning' : ''} />{refreshing ? 'Fetching data' : 'Fetch latest data'}</button></div></div>
      <div className="market-toolbar"><label><span>SHOW</span><select value={filter} onChange={event => updateView({ filter: event.target.value })}><option value="all">All five equities</option>{ASSETS.map(item => <option key={item.symbol} value={item.symbol}>{item.short}</option>)}</select></label><label><span>ORDER</span><select value={sort} onChange={event => updateView({ sort: event.target.value })}><option value="default">Research order</option><option value="high">Higher larger-move chance</option><option value="low">Lower larger-move chance</option></select></label><div className="market-segment" aria-label="Watchlist filter"><button type="button" className={starredOnly ? 'active' : ''} onClick={() => setStarredOnly(value => !value)} aria-pressed={starredOnly}>★ Starred{starred.length ? ` (${starred.length})` : ''}</button></div><span className="market-click-hint">Select a company to inspect price history and model inputs <span aria-hidden="true">↗</span></span></div>

      {serviceError ? <div className="market-service-error"><span className="market-error-code">API / 8000</span><div><strong>Market service unavailable</strong><p>{serviceError}</p><code>python -m src.prediction.server</code></div><button type="button" onClick={() => void loadSnapshot()}>Retry</button></div> : null}
      <div className="market-cards" aria-busy={loading}>
        {loading && !snapshot ? ASSETS.map(item => <div className="market-card-skeleton" key={item.symbol} />) : visibleAssets.map((item, index) => {
          const prediction = snapshot?.predictions[item.symbol]
          const failed = !prediction || Boolean(prediction.error)
          const larger = prediction?.prediction === 1
          const chance = prediction && !prediction.error ? predictedOutcome(prediction) : 0
          const delta = prediction?.latest_price_change_pct ?? 0
          const isStarred = starred.includes(item.symbol)
          return <div className={`market-card ${failed ? 'unavailable' : ''}`} key={item.symbol} style={{ '--card-order': index } as CSSProperties} role="button" tabIndex={failed ? -1 : 0} aria-disabled={failed} aria-label={failed ? `${item.name} outlook unavailable` : `Open ${item.name} stock details`} onClick={event => { if (!failed) openDetail(item.symbol, event.currentTarget) }} onKeyDown={event => { if (!failed && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); openDetail(item.symbol, event.currentTarget) } }}>
            <span className="market-card-top"><span className="market-symbol">{item.symbol}</span><button type="button" className="market-star" aria-pressed={isStarred} aria-label={isStarred ? `Remove ${item.symbol} from watchlist` : `Star ${item.symbol} to watchlist`} title={isStarred ? 'Starred' : 'Star'} onClick={event => { event.stopPropagation(); toggleStar(item.symbol) }}><Star size={14} fill={isStarred ? 'currentColor' : 'none'} /></button><span className={`market-freshness ${prediction?.stale ? 'stale' : failed ? 'waiting' : 'fresh'}`}><i />{failed ? 'NO FORECAST' : prediction.stale ? 'MODEL AGED' : 'MODEL READY'}</span></span>
            <span className="market-company">{item.name}</span>
            {failed ? <><span className="market-unavailable-title">Outlook unavailable</span><span className="market-error-detail">{prediction?.error ?? 'Market service has not returned a forecast yet.'}</span></> : <>
              <span className={`market-signal ${larger ? 'larger' : 'smaller'}`}><span className="market-signal-dot" /><span>{larger ? 'Larger moves' : 'Smaller moves'}</span><span className="market-signal-over">NEXT 60 MIN</span></span>
              <span className="market-chance-row"><strong>{Math.round(chance * 100)}<small>%</small></strong><span>model-estimated<br />chance</span></span>
              <span className="market-prob-track"><i className={larger ? 'larger' : 'smaller'} style={{ width: `${Math.round(chance * 100)}%` }} /></span>
              <span className="market-card-foot"><span>Last 15m <b className={delta >= 0 ? 'positive' : 'negative'}>{delta >= 0 ? '+' : ''}{delta.toFixed(2)}%</b></span><span>{ist(prediction.prediction_timestamp, true)} IST</span></span>
            </>}
          </div>
        })}
        {starredOnly && !visibleAssets.length && <div className="market-chart-empty">No starred equities yet — tap ★ on a card to build your watchlist.</div>}
      </div>
      {refreshMessage && <p className="market-refresh-message" role="status">{refreshMessage}</p>}
    </section>

    <section className="market-history" aria-labelledby="history-title">
      <div className="market-section-head"><div><p className="eyebrow">SHADOW RUN / RECENT RECORD</p><h2 id="history-title">Forecasts in time</h2></div><span className="market-history-count">{history.length ? `${history.length} recent entries` : 'No logged entries yet'}</span></div>
      {history.length ? <div className="market-table-scroll"><table className="market-history-table"><thead><tr><th>Company</th><th>Model call</th><th>Chance</th><th>Observed result</th><th>Forecast time</th></tr></thead><tbody>{history.map((entry, index) => {
        const result = entry.correct === null ? entry.outcome_status === 'not_scored' ? 'Not scorable' : entry.outcome_status === 'awaiting_data' ? 'Awaiting data' : entry.outcome_status === 'pending' ? 'In progress' : 'Pending' : entry.correct ? 'Correct' : 'Miss'
        const resultClass = entry.correct === null ? 'pending' : entry.correct ? 'hit' : 'miss'
        return <tr key={`${entry.asset}-${entry.prediction_timestamp}-${index}`}><td>{assetName(entry.asset)}</td><td>{entry.prediction ? 'Larger moves' : 'Smaller moves'}</td><td>{Math.round(entry.probability * 100)}%</td><td><span className={`market-result ${resultClass}`}>{result}</span></td><td>{ist(entry.prediction_timestamp, true)} IST</td></tr>
      })}</tbody></table></div> : <div className="market-history-empty"><span>NO SHADOW PREDICTIONS YET</span><p>Start the shadow runner to log hourly forecasts and score them when a complete outcome window is available.</p><code>python scripts/run_shadow.py</code></div>}
    </section>

    <section className="market-method-note"><div><p className="eyebrow">READING THE ESTIMATE</p><p>“Larger moves” means a wider high-to-low range than the asset’s training-period median. “Smaller moves” means a narrower range. This is about activity, not up or down.</p></div><div><p className="eyebrow">RESEARCH USE</p><p>Forecasts use completed 15-minute bars and a frozen walk-forward model. Historical patterns can fail; this page is a research prototype, not investment advice.</p></div></section>
    <footer className="market-footer"><span>ALPHASENSE / MARKET RESEARCH</span><span>NSE · IST · FIVE EQUITIES</span><a href="/#top" onClick={event => { event.preventDefault(); onNavigate('/#top') }}>RESEARCH NOTE ↑</a></footer>

    {detailAsset && <div className="market-dialog-backdrop" role="presentation" onMouseDown={event => { if (event.target === event.currentTarget) closeDetail() }}>
      <section className="market-dialog" ref={dialogRef} role="dialog" aria-modal="true" aria-labelledby="market-dialog-title">
        <div className="market-dialog-head"><div><p className="eyebrow">STOCK INSPECTION / {detailAsset}</p><h2 id="market-dialog-title">{assetName(detailAsset)}</h2><p>{selectedPrediction?.prediction_timestamp ? `Forecast timestamp · ${ist(selectedPrediction.prediction_timestamp, true)} IST` : 'Price history and model details'}</p></div><button type="button" className="market-dialog-close" aria-label="Close stock details" onClick={() => closeDetail()}><X size={17} /></button></div>
        <div className="market-detail-summary">{selectedPrediction?.error ? `Forecast unavailable: ${selectedPrediction.error}` : selectedPrediction ? <><strong>{selectedPrediction.prediction ? 'Larger' : 'Smaller'} price moves · {Math.round(predictedOutcome(selectedPrediction) * 100)}% model-estimated chance</strong><span>Observed last 15-minute price change: {change >= 0 ? '+' : ''}{change.toFixed(2)}%. The forecast estimates range magnitude, not direction.</span></> : <span>Forecast details are unavailable.</span>}</div>
        <section className="market-chart-section" aria-label="Price history chart"><div className="market-chart-heading"><div><span className="eyebrow">INTRADAY / PRICE HISTORY</span><strong>{chart?.bars.length ? price(chart.bars[chart.bars.length - 1].close) : '—'}</strong></div><span className="market-live-status"><i />{liveStatus}</span></div>
          <div className="market-chart-toolbar"><div className="market-segment" aria-label="Chart interval">{(['1m', '5m', '15m', '1h'] as const).map(value => <button key={value} type="button" className={interval === value ? 'active' : ''} onClick={() => updateView({ interval: value })}>{value}</button>)}</div><div className="market-segment" aria-label="Chart style">{(['candles', 'line'] as const).map(value => <button key={value} type="button" className={chartStyle === value ? 'active' : ''} onClick={() => setChartStyle(value)}>{value === 'candles' ? 'Candles' : 'Line'}</button>)}</div></div>
          {chartLoading ? <div className="market-chart-empty">Loading intraday bars…</div> : <SnapshotChart bars={chart?.bars ?? []} style={chartStyle} />}
          <p className="market-chart-caption">{chart?.message || 'Charts use intraday Upstox candles. Stream updates appear while the market feed is connected.'}</p>
        </section>
        <section className="market-chart-section" aria-label="Latest headlines"><div className="market-chart-heading"><div><span className="eyebrow">NEWS / LATEST HEADLINES</span><strong>{newsTotal ? `${newsTotal} stor${newsTotal === 1 ? 'y' : 'ies'}` : '—'}</strong></div><span className="market-live-status"><i />{newsRefreshAt ? `Updated ${ist(newsRefreshAt)} IST` : 'No news refresh yet'}</span></div>
          {newsLoading ? <div className="market-chart-empty">Loading headlines…</div> : news.length ? <><ul className="market-news-list">{news.map(item => <li key={item.url} className="market-news-item"><a href={item.url} target="_blank" rel="noreferrer">{item.headline}</a><span className="market-news-meta">{item.source} · {ist(item.published_at, true)} IST</span></li>)}</ul>
          {newsTotal > NEWS_PAGE_SIZE && <div className="market-news-pager"><button type="button" disabled={newsPage === 0 || newsLoading} onClick={() => setNewsPage(page => Math.max(0, page - 1))}>← Newer</button><span>Page {Math.min(newsPage + 1, newsPageCount)} of {newsPageCount}</span><button type="button" disabled={newsPage + 1 >= newsPageCount || newsLoading} onClick={() => setNewsPage(page => page + 1)}>Older →</button></div>}</> : <div className="market-chart-empty">No cached headlines yet — use Fetch latest data to pull news.</div>}
          <p className="market-chart-caption">Headlines are display-only from Upstox, Marketaux and GDELT. Predictions use market bars only.</p>
        </section>
        <details className="market-features"><summary>Advanced / standardized model inputs</summary><p>These values are standardized feature inputs, not rupee amounts or percentages. Inputs use completed history available at the forecast cutoff.</p>{selectedPrediction?.data_used?.features ? <div className="market-feature-table-wrap"><table className="market-feature-table"><thead><tr><th>Input</th><th>Standardized value</th></tr></thead><tbody>{FEATURE_GROUPS.map(group => <Fragment key={group.title}><tr className="market-feature-group"><td colSpan={2}>{group.title}</td></tr>{group.names.map(name => {
                const raw = selectedPrediction.data_used?.features?.[name]
                if (raw == null) return null
                const value = Number(raw)
                const sign = value > 0 ? 'positive' : value < 0 ? 'negative' : ''
                return <tr key={name}><td>{name.replaceAll('_', ' ')}</td><td className={sign}>{value > 0 ? '+' : ''}{value.toFixed(4)}</td></tr>
              })}</Fragment>)}</tbody></table></div> : <span>Feature details are unavailable.</span>}</details>
      </section>
    </div>}
  </main>
}
