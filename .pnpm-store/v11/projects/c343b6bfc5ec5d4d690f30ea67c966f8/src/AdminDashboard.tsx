import { useCallback, useEffect, useState, type CSSProperties } from 'react'
import {
  adminApi,
  downloadAdminCsv,
  type AdminAuditEntry,
  type AdminSession,
  type AdminShadowRow,
  type AdminUser,
  type AnalyticsOverview,
  type SystemConfig,
  type SystemStatus,
} from './auth-client'
import { AreaChart, Donut, HBar, Histogram, StackedBars, StatCard, useCountUp } from './admin-charts'
import './admin-dashboard.css'

type Tab = 'users' | 'sessions' | 'audit' | 'ops' | 'stats'

/* ---------------- small pieces ---------------- */

function initials(name: string) {
  const parts = name.trim().split(/\s+/)
  return ((parts[0]?.[0] ?? '?') + (parts.length > 1 ? (parts[1]?.[0] ?? '') : '')).toUpperCase()
}

function Beacon({ ok, label, mute }: { ok?: boolean; label: string; mute?: boolean }) {
  const cls = mute ? 'adm-beacon is-mute' : ok ? 'adm-beacon' : 'adm-beacon is-warn'
  return <span className={cls}><i aria-hidden="true" />{label}</span>
}

function Skeleton({ rows = 3 }: { rows?: number }) {
  return <div className="adm-skel" aria-hidden="true">{Array.from({ length: rows }).map((_, i) => <i key={i} />)}</div>
}

function Gauge({ index, label, ok, sub, countTo, format, text, meter }: {
  index: number; label: string; ok: boolean; sub: string;
  countTo?: number; format?: (n: number) => string; text?: string; meter?: number
}) {
  const n = useCountUp(countTo ?? 0)
  return (
    <article className={`adm-gauge${ok ? '' : ' is-warn'}`} style={{ '--i': index } as CSSProperties}>
      <span className="g-label"><Beacon ok={ok} label={label} /></span>
      <p className="g-value">{text ?? (format ? format(n) : n.toLocaleString('en-IN'))}</p>
      <span className="g-sub">{sub}</span>
      {meter != null && <span className="adm-meter" aria-hidden="true"><i style={{ width: `${Math.round(meter * 100)}%` }} /></span>}
    </article>
  )
}

/* ---------------- main ---------------- */

export function AdminDashboard() {
  const [tab, setTab] = useState<Tab>('users')
  const [users, setUsers] = useState<AdminUser[]>([])
  const [total, setTotal] = useState(0)
  const [search, setSearch] = useState('')
  const [includeDisabled, setIncludeDisabled] = useState(true)
  const [sessions, setSessions] = useState<AdminSession[]>([])
  const [audit, setAudit] = useState<AdminAuditEntry[]>([])
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [shadow, setShadow] = useState<AdminShadowRow[]>([])
  const [shadowCounts, setShadowCounts] = useState({ scored: 0, correct: 0, pending: 0 })
  const [shadowPage, setShadowPage] = useState(0)
  const [config, setConfig] = useState<SystemConfig | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const [busy, setBusy] = useState<string | null>(null)
  const [analytics, setAnalytics] = useState<AnalyticsOverview | null>(null)
  const [exporting, setExporting] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  const load = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      if (tab === 'users') {
        const data = await adminApi.listUsers({ search, include_disabled: includeDisabled })
        setUsers(data.users)
        setTotal(data.total)
      } else if (tab === 'sessions') {
        setSessions((await adminApi.listSessions()).sessions)
      } else if (tab === 'audit') {
        setAudit((await adminApi.listAudit()).entries)
      } else if (tab === 'stats') {
        setAnalytics(await adminApi.analytics())
      } else {
        const [statusData, shadowData, configData] = await Promise.all([
          adminApi.systemStatus(), adminApi.systemShadow(), adminApi.systemConfig(),
        ])
        setStatus(statusData)
        setShadow(shadowData.rows)
        setShadowCounts(shadowData.counts)
        setShadowPage(0)
        setConfig(configData)
      }
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Admin request failed.')
    } finally {
      setLoading(false)
    }
  }, [tab, search, includeDisabled])

  useEffect(() => {
    const timer = window.setTimeout(() => { void load() }, tab === 'users' ? 250 : 0)
    return () => window.clearTimeout(timer)
  }, [load, tab])

  async function mutate(key: string, action: () => Promise<unknown>, success: string) {
    setBusy(key)
    setError('')
    setNotice('')
    try {
      await action()
      setNotice(success)
      await load()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Admin request failed.')
    } finally {
      setBusy(null)
    }
  }

  async function refreshMarket() {
    setRefreshing(true)
    setError('')
    setNotice('Fetching latest market candles…')
    try {
      const result = await adminApi.systemRefresh()
      setNotice(`Updated ${result.assetsUpdated} stocks · ${result.predictionsAdded} new logged forecasts · ${result.outcomesScored} outcomes scored.`)
      await load()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Market refresh failed.')
    } finally {
      setRefreshing(false)
    }
  }

  async function exportCsv(kind: 'users' | 'audit' | 'shadow', path: '/api/admin/export/users' | '/api/admin/export/audit' | '/api/admin/export/shadow') {
    setExporting(kind)
    setError('')
    setNotice('')
    try {
      const filename = await downloadAdminCsv(path)
      setNotice(`Downloaded ${filename}.`)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Export failed.')
    } finally {
      setExporting(null)
    }
  }

  const tabs: { id: Tab; label: string; count?: number }[] = [
    { id: 'users', label: 'Users', count: total },
    { id: 'sessions', label: 'Sessions' },
    { id: 'audit', label: 'Audit log' },
    { id: 'ops', label: 'Ops' },
    { id: 'stats', label: 'Analytics' },
  ]
  const accuracy = shadowCounts.scored ? shadowCounts.correct / shadowCounts.scored : 0
  const SHADOW_PAGE_SIZE = 15
  const orderedShadow = shadow.slice().reverse()
  const shadowPageCount = Math.max(1, Math.ceil(orderedShadow.length / SHADOW_PAGE_SIZE))
  const safeShadowPage = Math.min(shadowPage, shadowPageCount - 1)
  const visibleShadow = orderedShadow.slice(safeShadowPage * SHADOW_PAGE_SIZE, safeShadowPage * SHADOW_PAGE_SIZE + SHADOW_PAGE_SIZE)

  return (
    <main id="market-main" tabIndex={-1} className="adm-page">
      <section className="adm-masthead" aria-labelledby="adm-title">
        <p className="adm-kicker"><span>OPERATIONS <b>/</b> ADMIN DESK</span><span>LOCAL ONLY</span></p>
        <h1 className="adm-title" id="adm-title">Run the <em>desk,</em><br />guard the ledger.</h1>
        <div className="adm-rule" aria-hidden="true" />
        <p className="adm-lede">
          <strong>Accounts, sessions, and system health</strong> for this research site.
          Promote, disable, revoke — every privileged change is written to the audit trail.
        </p>
        <div className="adm-meta">
          <span>NSE 09:15–15:30</span>
          <span className="adm-stamp">ADMIN CLEARANCE</span>
        </div>
      </section>

      <div className="adm-tabs" role="tablist" aria-label="Admin sections">
        {tabs.map((item, i) => (
          <button key={item.id} type="button" role="tab" aria-selected={tab === item.id}
            className="adm-tab" onClick={() => setTab(item.id)}>
            <span className="n">{String(i + 1).padStart(2, '0')}</span>
            {item.label}
            {item.count != null && item.count > 0 && <span className="count">{item.count}</span>}
          </button>
        ))}
      </div>

      {error && <p className="adm-error" role="alert">{error}</p>}
      {notice && <p className="adm-notice" role="status">{notice}</p>}

      <div className="adm-panel" key={tab}>
        {tab === 'users' && (
          <section aria-label="Users">
            <div className="adm-toolbar">
              <label className="adm-field">SEARCH THE LEDGER
                <input type="search" value={search} aria-label="Search users by email or name"
                  onChange={event => setSearch(event.target.value)} placeholder="email or name…" />
              </label>
              <label className="adm-check">
                <input type="checkbox" checked={includeDisabled} onChange={event => setIncludeDisabled(event.target.checked)} />
                INCLUDE DISABLED
              </label>
            </div>
            {loading ? <Skeleton rows={3} /> : users.length ? (
              <div className="adm-grid">
                {users.map((user, i) => {
                  const key = `user-${user.id}`
                  const acting = busy === key
                  return (
                    <article key={user.id} className={`adm-card${user.disabled ? ' is-disabled' : ''}`}
                      style={{ '--i': i } as CSSProperties} aria-label={`${user.displayName}, ${user.email}`}>
                      <div className="adm-card-top">
                        <span className="adm-avatar" aria-hidden="true">{initials(user.displayName)}</span>
                        <div className="adm-who">
                          <strong>{user.displayName}</strong>
                          <span title={user.email}>{user.email}</span>
                        </div>
                      </div>
                      <div className="adm-seals">
                        <span className={`adm-stamp${user.role === 'admin' ? '' : ' is-user'}`}>{user.role}</span>
                        <Beacon ok={!user.disabled} label={user.disabled ? 'Disabled' : 'Active'} />
                      </div>
                      <div className="adm-facts">
                        <div><span>USER ID</span><b>#{String(user.id).padStart(3, '0')}</b></div>
                        <div><span>MEMBER SINCE</span><b>{user.createdAt?.slice(0, 10) ?? '—'}</b></div>
                      </div>
                      <div className="adm-actions">
                        <button type="button" className="adm-btn adm-btn--sm" disabled={acting}
                          onClick={() => void mutate(key,
                            () => adminApi.updateUser(user.id, { role: user.role === 'admin' ? 'user' : 'admin' }),
                            `${user.email} is now ${user.role === 'admin' ? 'a user' : 'an admin'}.`)}>
                          {user.role === 'admin' ? 'Demote' : 'Promote'}
                        </button>
                        <button type="button" className={`adm-btn adm-btn--sm${user.disabled ? '' : ' adm-btn--danger'}`}
                          disabled={acting}
                          onClick={() => void mutate(key,
                            () => adminApi.updateUser(user.id, { disabled: !user.disabled }),
                            `${user.email} ${user.disabled ? 'enabled' : 'disabled; sessions revoked'}.`)}>
                          {user.disabled ? 'Enable' : 'Disable'}
                        </button>
                        <button type="button" className="adm-btn adm-btn--sm" disabled={acting}
                          onClick={() => void mutate(key,
                            () => adminApi.revokeUserSessions(user.id), `Sessions revoked for ${user.email}.`)}>
                          Revoke sessions
                        </button>
                      </div>
                    </article>
                  )
                })}
              </div>
            ) : (
              <div className="adm-empty"><strong>No users on this page.</strong><p>Try a different search, or include disabled accounts.</p></div>
            )}
          </section>
        )}

        {tab === 'sessions' && (
          <section aria-label="Sessions">
            {loading ? <Skeleton rows={3} /> : sessions.length ? (
              <div>
                {sessions.map((session, i) => (
                  <div key={session.tokenHash} className="adm-session" style={{ '--i': i } as CSSProperties}>
                    <span className="adm-hash" title={session.tokenHash}>{session.tokenHash.slice(0, 12)}…</span>
                    <div className="adm-session-meta">
                      <b>USER #{session.userId}</b>
                      <span>EXPIRES {session.expiresAt?.replace('T', ' ').slice(0, 16) ?? '—'}</span>
                    </div>
                    <Beacon ok label="Live" />
                    <button type="button" className="adm-btn adm-btn--sm adm-btn--danger" disabled={busy === session.tokenHash}
                      onClick={() => void mutate(session.tokenHash,
                        () => adminApi.revokeSession(session.tokenHash), 'Session revoked.')}>Revoke</button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="adm-empty"><strong>Quiet lines.</strong><p>No active sessions right now.</p></div>
            )}
          </section>
        )}

        {tab === 'audit' && (
          <section aria-label="Audit log">
            {loading ? <Skeleton rows={4} /> : audit.length ? (
              <div className="adm-timeline">
                {audit.map((entry, i) => {
                  const destructive = /revoke|disabled":true/.test(`${entry.action} ${entry.detail ?? ''}`)
                  return (
                    <div key={entry.id} className={`adm-entry${destructive ? ' is-destructive' : ''}`}
                      style={{ '--i': i } as CSSProperties}>
                      <article className="adm-entry-card">
                        <div className="adm-entry-top">
                          <code>{entry.action}</code>
                          <time>{entry.createdAt?.replace('T', ' ').slice(0, 16) ?? '—'}</time>
                        </div>
                        {entry.detail && <p className="adm-entry-detail">{entry.detail}</p>}
                        <p className="adm-entry-actors">ENTRY #{entry.id} · ACTOR #{entry.actorId}{entry.targetUserId != null && <> → TARGET #{entry.targetUserId}</>}</p>
                      </article>
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="adm-empty"><strong>Blank ledger.</strong><p>Privileged actions will appear here as they happen.</p></div>
            )}
          </section>
        )}

        {tab === 'ops' && (
          <section aria-label="Operations">
            <div className="adm-ops-head">
              <p>Same refresh flow as the market service — candles, outcome labels, shadow log. Audited as <b>system.refresh</b>.</p>
              <button type="button" className="adm-btn adm-btn--primary" onClick={() => void refreshMarket()} disabled={refreshing}>
                {refreshing && <span className="spin" aria-hidden="true" />}
                {refreshing ? 'Fetching data…' : 'Fetch latest data'}
              </button>
            </div>
            {loading ? <Skeleton rows={3} /> : status ? (<>
              <div className="adm-gauges">
                <Gauge index={0} label="Prediction API" ok={status.prediction.ok}
                  text={status.prediction.httpStatus != null ? String(status.prediction.httpStatus) : '—'}
                  sub={status.prediction.error ?? `feed ${status.prediction.feedState ?? 'unknown'}`} />
                <Gauge index={1} label="Market cache" ok={status.marketCache.present}
                  countTo={status.marketCache.rows ?? 0} format={n => n.toLocaleString('en-IN')}
                  sub={status.marketCache.present ? 'completed 15m bars' : 'start the shadow runner'} />
                <Gauge index={2} label="Shadow scored" ok={status.shadowLog.present}
                  countTo={shadowCounts.scored} sub={`${shadowCounts.correct} correct · ${shadowCounts.pending} pending`}
                  meter={accuracy} />
                <Gauge index={3} label="Artifacts" ok={status.modelArtifacts.ok}
                  countTo={status.modelArtifacts.present.length} format={n => `${n}/5`}
                  sub={status.modelArtifacts.ok ? 'all assets ready' : `missing ${status.modelArtifacts.missing.join(', ')}`} />
                <Gauge index={4} label="Upstox token" ok={status.upstoxTokenConfigured.ok}
                  text={status.upstoxTokenConfigured.ok ? 'SET' : '—'}
                  sub={status.upstoxTokenConfigured.ok ? 'presence only, never the value' : 'renew via OAuth'} />
                <Gauge index={5} label="Account DB" ok={status.authDb.ok}
                  text={status.authDb.ok ? 'LIVE' : 'DOWN'} sub={status.authDb.ok ? 'reachable' : 'check MySQL'} />
                <Gauge index={6} label="Web build" ok={status.webapp.ok}
                  text={status.webapp.ok ? 'BUILT' : '—'} sub={status.webapp.ok ? 'dist present' : 'run pnpm build'} />
              </div>
              {config && (
                <details className="adm-config">
                  <summary>Configuration — non-secret</summary>
                  <dl>
                    <div><dt>Assets</dt><dd className="mono">{config.assets.join(' · ')}</dd></div>
                    <div><dt>Session</dt><dd>{config.session} {config.timezone}</dd></div>
                    <div><dt>Horizon</dt><dd>{config.horizon} range-regime</dd></div>
                    <div><dt>Cache retention</dt><dd>{config.marketCacheRetentionDays} days</dd></div>
                    <div><dt>Model</dt><dd>{config.model}</dd></div>
                    <div><dt>Origins</dt><dd className="mono">{config.allowedOrigins.join(', ')}</dd></div>
                  </dl>
                </details>
              )}
              <div className="adm-section-title"><h3>Shadow forecasts</h3><span>{shadowCounts.scored} SCORED · {shadowCounts.correct} CORRECT · {shadowCounts.pending} PENDING</span></div>
              {shadow.length ? (<>
                <div className="adm-table-wrap"><table className="adm-table">
                  <thead><tr><th>Company</th><th>Forecast</th><th>Chance</th><th>Result</th><th>Time</th></tr></thead>
                  <tbody>{visibleShadow.map((row, i) => (
                    <tr key={`${row.asset}-${row.predictionTimestamp}-${i}`}>
                      <td><b>{row.asset ?? '—'}</b></td>
                      <td>{row.prediction === 1 ? 'Larger moves' : row.prediction === 0 ? 'Smaller moves' : '—'}</td>
                      <td className="mono">{row.probability == null ? '—' : `${Math.round(row.probability * 100)}%`}</td>
                      <td>{row.correct == null
                        ? <span className="adm-pill is-pending">Pending</span>
                        : row.correct
                          ? <span className="adm-pill">Correct</span>
                          : <span className="adm-pill is-miss">Miss</span>}</td>
                      <td className="mono">{row.predictionTimestamp?.replace('T', ' ').slice(0, 16) ?? '—'}</td>
                    </tr>
                  ))}</tbody>
                </table></div>
                {shadow.length > SHADOW_PAGE_SIZE && <div className="adm-pager"><button type="button" disabled={safeShadowPage === 0} onClick={() => setShadowPage(page => Math.max(0, page - 1))}>← Newer</button><span>Page {safeShadowPage + 1} of {shadowPageCount} · {shadow.length} rows</span><button type="button" disabled={safeShadowPage + 1 >= shadowPageCount} onClick={() => setShadowPage(page => page + 1)}>Older →</button></div>}
              </>) : <div className="adm-empty"><strong>Nothing logged yet.</strong><p>Start the shadow runner to record forecasts.</p></div>}
            </>) : <div className="adm-empty"><strong>Status unavailable.</strong><p>Reload the tab to retry.</p></div>}
          </section>
        )}

        {tab === 'stats' && (
          <section aria-label="Analytics">
            <div className="adm-ops-head">
              <p>Thirty-day account and shadow-model trends, computed on demand. Nothing here retrains, relabels, or touches research data.</p>
              <div className="adm-exports">
                <button type="button" className="adm-btn adm-btn--sm" disabled={exporting != null}
                  onClick={() => void exportCsv('users', '/api/admin/export/users')}>
                  {exporting === 'users' ? 'Preparing…' : 'Users CSV'}</button>
                <button type="button" className="adm-btn adm-btn--sm" disabled={exporting != null}
                  onClick={() => void exportCsv('audit', '/api/admin/export/audit')}>
                  {exporting === 'audit' ? 'Preparing…' : 'Audit CSV'}</button>
                <button type="button" className="adm-btn adm-btn--sm" disabled={exporting != null}
                  onClick={() => void exportCsv('shadow', '/api/admin/export/shadow')}>
                  {exporting === 'shadow' ? 'Preparing…' : 'Shadow CSV'}</button>
              </div>
            </div>
            {loading ? <Skeleton rows={4} /> : analytics ? (<>
              <div className="adm-gauges">
                <StatCard index={0} label="Total users" countTo={analytics.users.total}
                  format={n => n.toLocaleString('en-IN')}
                  sub={`${analytics.users.active} active · ${analytics.users.disabled} disabled`} />
                <StatCard index={1} label="Live sessions" countTo={analytics.sessions.live}
                  format={n => n.toLocaleString('en-IN')}
                  sub={`avg ${analytics.sessions.avgPerUser} per user`} />
                <StatCard index={2} label="Shadow accuracy" countTo={Math.round((analytics.shadow.accuracy ?? 0) * 100)}
                  format={n => analytics.shadow.accuracy == null ? '—' : `${n}%`}
                  sub={`${analytics.shadow.correct}/${analytics.shadow.scored} scored · ${analytics.shadow.pending} pending`} />
                <StatCard index={3} label="Admin actions · 24h" countTo={analytics.audit.last24h}
                  format={n => n.toLocaleString('en-IN')} sub="across all actors" />
              </div>
              <div className="adm-charts">
                <article className="adm-chart-card" style={{ '--i': 0 } as CSSProperties}>
                  <h4>Signups per day</h4>
                  <p className="adm-chart-sub">New accounts · last 30 days (IST)</p>
                  <AreaChart label="Signups per day" format={n => `${n}`}
                    points={analytics.users.signupsPerDay.map(d => ({ date: d.date, value: d.count }))} />
                </article>
                <article className="adm-chart-card" style={{ '--i': 1 } as CSSProperties}>
                  <h4>Sessions created</h4>
                  <p className="adm-chart-sub">Sign-ins per day · last 30 days (IST)</p>
                  <AreaChart label="Sessions created per day" format={n => `${n}`}
                    points={analytics.sessions.perDay.map(d => ({ date: d.date, value: d.count }))} />
                </article>
                <article className="adm-chart-card adm-chart-wide" style={{ '--i': 2 } as CSSProperties}>
                  <h4>Admin activity</h4>
                  <p className="adm-chart-sub">Audit actions per day by family · last 30 days (IST)</p>
                  <StackedBars label="Admin activity per day" days={analytics.audit.perDay} />
                  {analytics.audit.topActors.length > 0 && (
                    <ul className="adm-topactors">
                      {analytics.audit.topActors.map(actor => (
                        <li key={actor.actorId}>
                          <b>{actor.email ?? `USER #${actor.actorId}`}</b>
                          <span>{actor.count} action{actor.count === 1 ? '' : 's'}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </article>
                <article className="adm-chart-card" style={{ '--i': 3 } as CSSProperties}>
                  <h4>Account mix</h4>
                  <p className="adm-chart-sub">Who holds access right now</p>
                  <Donut label="Account mix" segments={[
                    { label: 'admins', value: analytics.users.admins, color: 'var(--teal)' },
                    { label: 'users', value: analytics.users.active - analytics.users.admins, color: 'var(--ink)' },
                    { label: 'disabled', value: analytics.users.disabled, color: 'var(--muted)' },
                  ]} />
                </article>
                <article className="adm-chart-card" style={{ '--i': 4 } as CSSProperties}>
                  <h4>Hit rate by asset</h4>
                  <p className="adm-chart-sub">Scored shadow forecasts only</p>
                  {analytics.shadow.perAsset.length ? analytics.shadow.perAsset.map((row, i) => (
                    <HBar key={row.asset} index={i} label={row.asset}
                      fraction={row.accuracy ?? 0} tone={row.accuracy != null && row.accuracy < 0.5 ? 'warn' : undefined}
                      display={row.accuracy == null ? `0/${row.scored}` : `${Math.round(row.accuracy * 100)}% · ${row.correct}/${row.scored}`} />
                  )) : <p className="adm-chart-sub">No scored forecasts yet.</p>}
                </article>
                <article className="adm-chart-card" style={{ '--i': 5 } as CSSProperties}>
                  <h4>Model confidence</h4>
                  <p className="adm-chart-sub">Forecast probability histogram · is the model decisive?</p>
                  <Histogram label="Forecast confidence" buckets={analytics.shadow.confidence} />
                </article>
                <article className="adm-chart-card adm-chart-wide" style={{ '--i': 6 } as CSSProperties}>
                  <h4>Rolling 7-day accuracy</h4>
                  <p className="adm-chart-sub">Share correct over the trailing 7 scored days · gaps mean no scored forecasts</p>
                  <AreaChart label="Rolling 7-day accuracy" format={n => `${Math.round(n * 100)}%`}
                    points={analytics.shadow.rolling7d.map(d => ({ date: d.date, value: d.accuracy }))} />
                </article>
              </div>
            </>) : <div className="adm-empty"><strong>No analytics yet.</strong><p>Reload the tab to retry.</p></div>}
          </section>
        )}
      </div>

      <p className="adm-foot">ALPHASENSE ADMIN DESK · EVERY PRIVILEGED CHANGE IS WRITTEN TO THE AUDIT TRAIL</p>
    </main>
  )
}
