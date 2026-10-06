import { useCallback, useEffect, useState, type FormEvent } from 'react'
import { ArrowRight, ArrowUpRight, KeyRound, Mail, MessageCircle, ShieldCheck, Star, UserRound } from 'lucide-react'
import { changePassword, getNotificationPrefs, updateDisplayName, updateEmail, updateNotificationPrefs, type AuthUser } from './auth-client'
import './profile.css'

type ProfilePageProps = {
  user: AuthUser
  onNavigate: (path: string) => void
  onUserChange: (user: AuthUser) => void
}

const ASSETS = [
  { symbol: 'RELIANCE', name: 'Reliance Industries', short: 'Reliance' },
  { symbol: 'TCS', name: 'Tata Consultancy Services', short: 'TCS' },
  { symbol: 'HDFCBANK', name: 'HDFC Bank', short: 'HDFC Bank' },
  { symbol: 'INFY', name: 'Infosys', short: 'Infosys' },
  { symbol: 'ICICIBANK', name: 'ICICI Bank', short: 'ICICI Bank' },
] as const

type Asset = typeof ASSETS[number]['symbol']
type Prediction = {
  prediction_timestamp: string
  probability_high_range: number
  prediction: 0 | 1
  latest_price_change_pct: number
  stale?: boolean
  error?: string
}
type HistoryItem = { asset: string; prediction_timestamp: string; probability: number; prediction: 0 | 1; correct: boolean | null; outcome_status: string }
type Snapshot = { predictions: Partial<Record<Asset, Prediction>>; history: HistoryItem[]; checked_at: string }

const MARKET_API = (import.meta.env.VITE_MARKET_API_BASE_URL || '/market-api').replace(/\/$/, '')
const WATCHLIST_KEY = 'alphasense-watchlist'

function readWatchlist(): Asset[] {
  try {
    const saved = JSON.parse(window.localStorage.getItem(WATCHLIST_KEY) ?? '[]') as string[]
    return ASSETS.map(item => item.symbol).filter((symbol): symbol is Asset => saved.includes(symbol))
  } catch {
    return []
  }
}

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

function chanceOf(prediction: Prediction) {
  return prediction.prediction === 1 ? prediction.probability_high_range : 1 - prediction.probability_high_range
}

function initialsOf(name: string) {
  const parts = name.trim().split(/\s+/).filter(Boolean)
  if (!parts.length) return '·'
  return (parts[0][0] + (parts.length > 1 ? parts[parts.length - 1][0] : '')).toUpperCase()
}

function fileNoOf(email: string) {
  let hash = 0
  for (let i = 0; i < email.length; i++) hash = (hash * 31 + email.charCodeAt(i)) >>> 0
  return String(hash % 9000 + 1000)
}

export function ProfilePage({ user, onNavigate, onUserChange }: ProfilePageProps) {
  const [watchlist, setWatchlist] = useState<Asset[]>(readWatchlist)
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [marketsDown, setMarketsDown] = useState(false)
  const [name, setName] = useState(user.displayName)
  const [nameBusy, setNameBusy] = useState(false)
  const [nameError, setNameError] = useState('')
  const [nameSaved, setNameSaved] = useState(false)
  const [emailAddress, setEmailAddress] = useState(user.email)
  const [emailBusy, setEmailBusy] = useState(false)
  const [emailError, setEmailError] = useState('')
  const [emailSaved, setEmailSaved] = useState(false)
  const [passwordBusy, setPasswordBusy] = useState(false)
  const [passwordError, setPasswordError] = useState('')
  const [passwordSaved, setPasswordSaved] = useState(false)
  const [showPasswords, setShowPasswords] = useState(false)
  const [notifLoading, setNotifLoading] = useState(true)
  const [emailAlerts, setEmailAlerts] = useState(true)
  const [whatsappAlerts, setWhatsappAlerts] = useState(false)
  const [phone, setPhone] = useState('')
  const [notifBusy, setNotifBusy] = useState(false)
  const [notifError, setNotifError] = useState('')
  const [notifSaved, setNotifSaved] = useState(false)

  // Stay in sync when the watchlist changes in the Markets tab.
  useEffect(() => {
    const sync = (event: StorageEvent) => { if (event.key === WATCHLIST_KEY) setWatchlist(readWatchlist()) }
    window.addEventListener('storage', sync)
    return () => window.removeEventListener('storage', sync)
  }, [])

  useEffect(() => {
    let active = true
    fetch(`${MARKET_API}/dashboard`)
      .then(async response => {
        if (!response.ok) throw new Error(`Market service returned ${response.status}.`)
        const data = await response.json() as Snapshot
        if (active) {
          setSnapshot(data)
          setMarketsDown(false)
        }
      })
      .catch(() => { if (active) setMarketsDown(true) })
    return () => { active = false }
  }, [])

  useEffect(() => {
    setName(user.displayName)
  }, [user.displayName])

  useEffect(() => {
    setEmailAddress(user.email)
  }, [user.email])

  useEffect(() => {
    let active = true
    getNotificationPrefs()
      .then(prefs => {
        if (!active) return
        setEmailAlerts(prefs.emailEnabled)
        setWhatsappAlerts(prefs.whatsappEnabled)
        setPhone(prefs.whatsappNumber ?? '')
      })
      .catch(() => { /* prefs stay at defaults; save will surface errors */ })
      .finally(() => { if (active) setNotifLoading(false) })
    return () => { active = false }
  }, [])

  const toggleStar = useCallback((asset: Asset) => {
    setWatchlist(current => {
      const next = current.includes(asset) ? current.filter(symbol => symbol !== asset) : [...current, asset]
      try {
        window.localStorage.setItem(WATCHLIST_KEY, JSON.stringify(next))
      } catch { /* private mode: watchlist lasts for this tab */ }
      return next
    })
  }, [])

  const go = (event: React.MouseEvent, path: string) => { event.preventDefault(); onNavigate(path) }
  const firstName = user.displayName.split(' ')[0] || user.displayName
  const isAdmin = user.role === 'admin'
  const watched = ASSETS.filter(item => watchlist.includes(item.symbol))
  const allHistory = snapshot?.history ?? []
  const history = allHistory.slice(0, 5)
  const hits = allHistory.filter(entry => entry.correct === true).length

  async function submitName(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setNameError('')
    setNameSaved(false)
    const trimmed = name.trim()
    if (!trimmed) {
      setNameError('Enter your name.')
      return
    }
    setNameBusy(true)
    try {
      const updated = await updateDisplayName(trimmed)
      onUserChange(updated)
      setName(updated.displayName)
      setNameSaved(true)
    } catch (reason) {
      setNameError(reason instanceof Error ? reason.message : 'We could not save that name. Please try again.')
    } finally {
      setNameBusy(false)
    }
  }

  async function submitEmail(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setEmailError('')
    setEmailSaved(false)
    const trimmed = emailAddress.trim()
    if (!trimmed) {
      setEmailError('Enter your email address.')
      return
    }
    if (trimmed.toLowerCase() === user.email.toLowerCase()) {
      setEmailError('That is already your sign-in email.')
      return
    }
    setEmailBusy(true)
    try {
      const updated = await updateEmail(trimmed)
      onUserChange(updated)
      setEmailAddress(updated.email)
      setEmailSaved(true)
    } catch (reason) {
      setEmailError(reason instanceof Error ? reason.message : 'We could not save that email. Please try again.')
    } finally {
      setEmailBusy(false)
    }
  }

  async function submitPassword(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setPasswordError('')
    setPasswordSaved(false)
    const form = new FormData(event.currentTarget)
    const current = String(form.get('currentPassword') || '')
    const next = String(form.get('newPassword') || '')
    if (next !== String(form.get('confirmPassword') || '')) {
      setPasswordError('Those new passwords do not match.')
      return
    }
    setPasswordBusy(true)
    try {
      await changePassword(current, next)
      setPasswordSaved(true)
      event.currentTarget.reset()
    } catch (reason) {
      setPasswordError(reason instanceof Error ? reason.message : 'We could not change that password. Please try again.')
    } finally {
      setPasswordBusy(false)
    }
  }

  async function submitNotifications(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setNotifError('')
    setNotifSaved(false)
    const cleaned = phone.replace(/[\s\-()]/g, '').trim()
    if (whatsappAlerts && !cleaned) {
      setNotifError('Add your WhatsApp number to enable WhatsApp reminders.')
      return
    }
    setNotifBusy(true)
    try {
      const updated = await updateNotificationPrefs({
        email_enabled: emailAlerts,
        whatsapp_enabled: whatsappAlerts,
        whatsapp_e164: cleaned ? cleaned : null,
      })
      setEmailAlerts(updated.emailEnabled)
      setWhatsappAlerts(updated.whatsappEnabled)
      setPhone(updated.whatsappNumber ?? '')
      setNotifSaved(true)
    } catch (reason) {
      setNotifError(reason instanceof Error ? reason.message : 'We could not save those preferences. Please try again.')
    } finally {
      setNotifBusy(false)
    }
  }

  return <main className="pf-page" id="main" tabIndex={-1}>
    {/* ── Dossier header ─────────────────────────────────────────── */}
    <header className="pf-dossier">
      <div className="pf-dossier-top">
        <p className="pf-eyebrow">ALPHASENSE / MEMBER FILE</p>
        <span className="pf-file-no">FILE Nº A-{fileNoOf(user.email)}</span>
      </div>
      <div className="pf-dossier-main">
        <div className="pf-identity">
          <span className="pf-seal" aria-hidden="true">{initialsOf(user.displayName)}</span>
          <div>
            <h1>Hello, <em>{firstName}</em>.</h1>
            <p className="pf-identity-line">
              <Mail size={12} aria-hidden="true" />{user.email}
              <span className={`pf-role${isAdmin ? ' pf-role--admin' : ''}`}><i aria-hidden="true" />{isAdmin ? 'ADMIN' : 'RESEARCH ACCESS'}</span>
            </p>
          </div>
        </div>
        <div className="pf-actions">
          <a className="pf-cta" href="/markets" onClick={event => go(event, '/markets')}>Open Markets<ArrowUpRight size={14} /></a>
        </div>
      </div>
      <dl className="pf-stats" aria-label="Account at a glance">
        <div><span className="pf-stat-num">{watchlist.length}</span><span className="pf-stat-label">Starred equities</span></div>
        <div><span className="pf-stat-num">{allHistory.length}</span><span className="pf-stat-label">Forecasts logged</span></div>
        <div><span className="pf-stat-num">{hits}</span><span className="pf-stat-label">Confirmed hits</span></div>
      </dl>
    </header>

    {/* ── Watchlist ──────────────────────────────────────────────── */}
    <section aria-labelledby="pf-watchlist-heading">
      <div className="pf-section-head">
        <div><p className="pf-eyebrow">YOUR WATCHLIST</p><h2 id="pf-watchlist-heading">Starred equities</h2></div>
        <span className="pf-count">{watchlist.length ? `${watchlist.length} STARRED` : 'NONE YET'}</span>
      </div>
      {watched.length ? <div className="pf-cards">{watched.map((item, index) => {
        const prediction = snapshot?.predictions[item.symbol]
        const ok = prediction && !prediction.error
        const larger = ok && prediction.prediction === 1
        const delta = prediction?.latest_price_change_pct ?? 0
        return <article className="pf-card" key={item.symbol}>
          <div className="pf-card-head">
            <span className="pf-index">{String(index + 1).padStart(2, '0')}</span>
            <button type="button" className="pf-star" aria-pressed="true" aria-label={`Remove ${item.symbol} from watchlist`} title="Starred — remove" onClick={() => toggleStar(item.symbol)}><Star size={15} fill="currentColor" /></button>
          </div>
          <p className="pf-symbol">{item.symbol}</p>
          <h3>{item.name}</h3>
          {ok
            ? <p className={`pf-signal ${larger ? 'pf-signal--larger' : 'pf-signal--smaller'}`}><i aria-hidden="true" />{larger ? 'Larger moves' : 'Smaller moves'} · <b>{Math.round(chanceOf(prediction) * 100)}%</b></p>
            : <p className="pf-muted">{marketsDown ? 'Market service unreachable.' : 'No forecast yet.'}</p>}
          {ok && <div className="pf-card-foot">
            <span>Last 15m <span className={delta >= 0 ? 'positive' : 'negative'}>{delta >= 0 ? '+' : ''}{delta.toFixed(2)}%</span></span>
            <span>{ist(prediction.prediction_timestamp)} IST</span>
          </div>}
          <a className="pf-card-link" href={`/markets?stock=${item.symbol}`} onClick={event => go(event, `/markets?stock=${item.symbol}`)}>Inspect <ArrowRight size={13} /></a>
        </article>
      })}</div> : <div className="pf-empty">
        <Star size={30} className="pf-empty-star" aria-hidden="true" />
        <div><strong>No starred equities</strong><p>Tap ★ on any Markets card and it lands here with its latest outlook.</p></div>
        <a className="pf-cta" href="/markets" onClick={event => go(event, '/markets')}>Browse Markets<ArrowUpRight size={14} /></a>
      </div>}
    </section>

    {/* ── Ledger ─────────────────────────────────────────────────── */}
    <section aria-labelledby="pf-activity-heading">
        <div className="pf-section-head">
          <div><p className="pf-eyebrow">SHADOW RUN</p><h2 id="pf-activity-heading">Recent forecasts</h2></div>
        </div>
        {history.length ? <ul className="pf-ledger">{history.map((entry, index) => {
          const pill = entry.correct === null ? 'pf-pill--pending' : entry.correct ? 'pf-pill--hit' : 'pf-pill--miss'
          const result = entry.correct === null ? 'Pending' : entry.correct ? 'Hit' : 'Miss'
          return <li key={`${entry.asset}-${entry.prediction_timestamp}-${index}`}>
            <span className="pf-ledger-asset">{entry.asset}</span>
            <span className="pf-ledger-call">{entry.prediction ? 'Larger' : 'Smaller'} · {Math.round(entry.probability * 100)}%</span>
            <span className={`pf-pill ${pill}`}>{result}</span>
            <span className="pf-ledger-time">{ist(entry.prediction_timestamp, true)} IST</span>
          </li>
        })}</ul> : <p className="pf-muted">{marketsDown ? 'Market service is unreachable right now.' : 'No logged forecasts yet.'}</p>}
      </section>

    {/* ── Settings ───────────────────────────────────────────────── */}
    <section aria-labelledby="pf-settings-heading">
      <div className="pf-section-head">
        <div><p className="pf-eyebrow">ACCOUNT SETTINGS</p><h2 id="pf-settings-heading">Name, email & password</h2></div>
      </div>
      <div className="pf-settings-grid">
        <form className="pf-setting-card" onSubmit={submitName} aria-label="Change display name">
          <h3><UserRound size={18} aria-hidden="true" />Display name</h3>
          <p>Shown across the workspace and the top bar.</p>
          <label className="pf-field"><span>Name</span><input value={name} onChange={event => setName(event.target.value)} autoComplete="name" required maxLength={80} placeholder="Your name" /></label>
          {nameError && <p className="pf-error" role="alert">{nameError}</p>}
          {nameSaved && <p className="pf-success" role="status">Name saved.</p>}
          <button className="pf-submit" type="submit" disabled={nameBusy}>{nameBusy ? 'Saving…' : 'Save name'}<ArrowRight size={14} /></button>
        </form>
        <form className="pf-setting-card" onSubmit={submitEmail} aria-label="Change email address">
          <h3><Mail size={18} aria-hidden="true" />Email</h3>
          <p>Used to sign in and receive regime reminders.</p>
          <label className="pf-field"><span>Email</span><input value={emailAddress} onChange={event => setEmailAddress(event.target.value)} type="email" autoComplete="email" required maxLength={254} placeholder="you@example.com" /></label>
          {emailError && <p className="pf-error" role="alert">{emailError}</p>}
          {emailSaved && <p className="pf-success" role="status">Email updated. Use it next time you sign in.</p>}
          <button className="pf-submit" type="submit" disabled={emailBusy}>{emailBusy ? 'Saving…' : 'Save email'}<ArrowRight size={14} /></button>
        </form>
        <form className="pf-setting-card" onSubmit={submitPassword} aria-label="Change password">
          <h3><KeyRound size={18} aria-hidden="true" />Password</h3>
          <p>At least 10 characters. You stay signed in on this device.</p>
          <label className="pf-field"><span>Current password</span><input name="currentPassword" type={showPasswords ? 'text' : 'password'} autoComplete="current-password" required maxLength={128} placeholder="Your current password" /></label>
          <label className="pf-field"><span>New password</span><input name="newPassword" type={showPasswords ? 'text' : 'password'} autoComplete="new-password" required minLength={10} maxLength={128} placeholder="At least 10 characters" /></label>
          <label className="pf-field"><span>Confirm new password</span><input name="confirmPassword" type={showPasswords ? 'text' : 'password'} autoComplete="new-password" required minLength={10} maxLength={128} placeholder="Enter it once more" /></label>
          <label className="pf-check"><input type="checkbox" checked={showPasswords} onChange={event => setShowPasswords(event.target.checked)} /> Show passwords</label>
          {passwordError && <p className="pf-error" role="alert">{passwordError}</p>}
          {passwordSaved && <p className="pf-success" role="status">Password changed.</p>}
          <button className="pf-submit" type="submit" disabled={passwordBusy}>{passwordBusy ? 'Please wait…' : 'Change password'}<ArrowRight size={14} /></button>
        </form>
      </div>
    </section>

    {/* ── Notifications ──────────────────────────────────────────── */}
    <section aria-labelledby="pf-notif-heading">
      <div className="pf-section-head">
        <div><p className="pf-eyebrow">NOTIFICATIONS</p><h2 id="pf-notif-heading">WhatsApp & alerts</h2></div>
        <span className="pf-count">{notifLoading ? 'LOADING' : whatsappAlerts ? 'WHATSAPP ON' : 'WHATSAPP OFF'}</span>
      </div>
      <div className="pf-settings-grid">
        <form className="pf-setting-card" onSubmit={submitNotifications} aria-label="WhatsApp and alert preferences">
          <h3><MessageCircle size={18} aria-hidden="true" />Reminder preferences</h3>
          <p>High-range outlooks during 09:15–15:30 IST. WhatsApp needs a number in E.164 format.</p>
          <label className="pf-field"><span>WhatsApp number</span><input value={phone} onChange={event => setPhone(event.target.value)} autoComplete="tel" inputMode="tel" maxLength={20} placeholder="+919876543210" aria-label="WhatsApp number in E.164 format" /></label>
          <label className="pf-check"><input type="checkbox" checked={whatsappAlerts} onChange={event => setWhatsappAlerts(event.target.checked)} /> WhatsApp reminders</label>
          <label className="pf-check"><input type="checkbox" checked={emailAlerts} onChange={event => setEmailAlerts(event.target.checked)} /> Email reminders</label>
          {notifError && <p className="pf-error" role="alert">{notifError}</p>}
          {notifSaved && <p className="pf-success" role="status">Preferences saved.</p>}
          <button className="pf-submit" type="submit" disabled={notifBusy || notifLoading}>{notifBusy ? 'Saving…' : 'Save preferences'}<ArrowRight size={14} /></button>
        </form>
      </div>
    </section>

    <div className="pf-bottomline"><span>RESEARCH PROTOTYPE · NOT TRADING ADVICE</span><span><ShieldCheck size={11} aria-hidden="true" style={{ verticalAlign: '-1px' }} /> ACCOUNT SERVICE / MYSQL</span></div>
  </main>
}
