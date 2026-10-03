/* Hand-rolled SVG/div charts for the admin Analytics tab.
   No chart library — keeps the admin chunk light and the look editorial. */

import { useEffect, useId, useState, type CSSProperties } from 'react'

export function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(
    () => typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  )
  useEffect(() => {
    const query = window.matchMedia('(prefers-reduced-motion: reduce)')
    const update = () => setReduced(query.matches)
    query.addEventListener('change', update)
    return () => query.removeEventListener('change', update)
  }, [])
  return reduced
}

export function useCountUp(target: number, duration = 950) {
  const reduced = usePrefersReducedMotion()
  const [value, setValue] = useState(0)
  useEffect(() => {
    if (reduced) {
      setValue(target)
      return
    }
    let raf = 0
    const start = performance.now()
    const tick = (now: number) => {
      const progress = Math.min(1, (now - start) / duration)
      setValue(Math.round(target * (1 - Math.pow(1 - progress, 3))))
      if (progress < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [target, duration, reduced])
  return value
}

/* ---------- area / line chart ---------- */

export function AreaChart({ points, format, label }: {
  points: { date: string; value: number | null }[]
  format: (n: number) => string
  label: string
}) {
  const gradientId = useId()
  const W = 320, H = 110, PAD_L = 34, PAD_B = 18, PAD_T = 10
  const values = points.map(p => p.value).filter((v): v is number => v != null)
  const max = Math.max(1, ...values)
  const x = (i: number) => PAD_L + (i / Math.max(1, points.length - 1)) * (W - PAD_L - 8)
  const y = (v: number) => PAD_T + (1 - v / max) * (H - PAD_T - PAD_B)
  // Split into contiguous non-null runs so gaps render as gaps.
  const runs: { run: { date: string; value: number }[]; start: number }[] = []
  points.forEach((p, i) => {
    if (p.value == null) return
    const last = runs[runs.length - 1]
    if (last && last.start + last.run.length === i) last.run.push({ date: p.date, value: p.value })
    else runs.push({ run: [{ date: p.date, value: p.value }], start: i })
  })
  const summary = values.length
    ? `range ${format(Math.min(...values))} to ${format(max)}`
    : 'no data'
  return (
    <svg className="adm-area" viewBox={`0 0 ${W} ${H}`} role="img"
      aria-label={`${label}: ${summary}`}>
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="var(--accent)" stopOpacity=".28" />
          <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[0.25, 0.5, 0.75, 1].map(frac => (
        <g key={frac}>
          <line x1={PAD_L} x2={W - 4} y1={PAD_T + frac * (H - PAD_T - PAD_B)} y2={PAD_T + frac * (H - PAD_T - PAD_B)} className="adm-area-grid" />
          {frac === 1 && <text x={2} y={PAD_T + frac * (H - PAD_T - PAD_B) + 3} className="adm-area-axis">{format(max)}</text>}
        </g>
      ))}
      {runs.map(({ run, start }, r) => {
        const line = run.map((p, k) => `${k === 0 ? 'M' : 'L'}${x(start + k).toFixed(1)},${y(p.value).toFixed(1)}`).join(' ')
        const base = H - PAD_B
        const area = `${line} L${x(start + run.length - 1).toFixed(1)},${base} L${x(start).toFixed(1)},${base} Z`
        const last = run[run.length - 1]
        return (
          <g key={r}>
            <path d={area} fill={`url(#${gradientId})`} className="adm-area-fill" />
            <path d={line} fill="none" className="adm-area-line">
              <title>{`${label}: ${summary}`}</title>
            </path>
            <circle cx={x(start + run.length - 1)} cy={y(last.value)} r={3.5} className="adm-area-dot" />
          </g>
        )
      })}
      <text x={PAD_L} y={H - 4} className="adm-area-axis">{points[0]?.date.slice(5) ?? ''}</text>
      <text x={W - 4} y={H - 4} textAnchor="end" className="adm-area-axis">{points[points.length - 1]?.date.slice(5) ?? ''}</text>
    </svg>
  )
}

/* ---------- stacked daily bars (audit families) ---------- */

const FAMILY_COLORS = { user: 'var(--teal)', session: 'var(--accent)', system: 'var(--ink)', other: 'var(--hairline)' } as const

export function StackedBars({ days, label }: {
  days: { date: string; user: number; session: number; system: number; other: number }[]
  label: string
}) {
  const max = Math.max(1, ...days.map(d => d.user + d.session + d.system + d.other))
  const total = days.reduce((n, d) => n + d.user + d.session + d.system + d.other, 0)
  return (
    <div className="adm-stack" role="img" aria-label={`${label}: ${total} actions in 30 days`}>
      <div className="adm-stack-bars">
        {days.map(day => {
          const sum = day.user + day.session + day.system + day.other
          return (
            <span key={day.date} className="adm-stack-col" title={`${day.date}: ${sum}`}
              style={{ height: `${Math.max(3, (sum / max) * 100)}%` }}>
              {(Object.keys(FAMILY_COLORS) as (keyof typeof FAMILY_COLORS)[]).map(family => (
                day[family] > 0 && (
                  <i key={family} style={{ flexGrow: day[family], background: FAMILY_COLORS[family] }} />
                )
              ))}
            </span>
          )
        })}
      </div>
      <div className="adm-stack-key">
        <span><i style={{ background: FAMILY_COLORS.user }} />user</span>
        <span><i style={{ background: FAMILY_COLORS.session }} />session</span>
        <span><i style={{ background: FAMILY_COLORS.system }} />system</span>
      </div>
    </div>
  )
}

/* ---------- horizontal bar ---------- */

export function HBar({ label, display, fraction, tone, index }: {
  label: string; display: string; fraction: number; tone?: 'warn'; index: number
}) {
  const [on, setOn] = useState(false)
  useEffect(() => {
    const raf = requestAnimationFrame(() => setOn(true))
    return () => cancelAnimationFrame(raf)
  }, [])
  return (
    <div className="adm-hbar" style={{ animationDelay: `${index * 60}ms` }}>
      <span className="adm-hbar-label">{label}</span>
      <span className="adm-hbar-track">
        <i className={tone === 'warn' ? 'is-warn' : ''} style={{ width: on ? `${Math.round(fraction * 100)}%` : '0%' }} />
      </span>
      <span className="adm-hbar-value">{display}</span>
    </div>
  )
}

/* ---------- donut ---------- */

export function Donut({ segments, label }: {
  segments: { label: string; value: number; color: string }[]
  label: string
}) {
  const total = segments.reduce((n, s) => n + s.value, 0)
  const R = 44, C = 2 * Math.PI * R
  let offset = 0
  return (
    <div className="adm-donut" role="img" aria-label={`${label}: ${segments.map(s => `${s.label} ${s.value}`).join(', ')}`}>
      <svg viewBox="0 0 120 120" className="adm-donut-svg">
        <circle cx={60} cy={60} r={R} className="adm-donut-bg" />
        {segments.map(seg => {
          const frac = total ? seg.value / total : 0
          const el = (
            <circle key={seg.label} cx={60} cy={60} r={R} fill="none"
              stroke={seg.color} strokeWidth={15}
              strokeDasharray={`${(frac * C).toFixed(1)} ${C.toFixed(1)}`}
              strokeDashoffset={(-offset * C).toFixed(1)}
              transform="rotate(-90 60 60)" strokeLinecap="butt" />
          )
          offset += frac
          return el
        })}
        <text x={60} y={58} textAnchor="middle" className="adm-donut-total">{total}</text>
        <text x={60} y={72} textAnchor="middle" className="adm-donut-sub">total</text>
      </svg>
      <div className="adm-donut-key">
        {segments.map(seg => (
          <span key={seg.label}><i style={{ background: seg.color }} />{seg.label} · <b>{seg.value}</b></span>
        ))}
      </div>
    </div>
  )
}

/* ---------- histogram ---------- */

export function Histogram({ buckets, label }: { buckets: number[]; label: string }) {
  const max = Math.max(1, ...buckets)
  const total = buckets.reduce((n, b) => n + b, 0)
  return (
    <div className="adm-hist" role="img" aria-label={`${label}: ${total} forecasts`}>
      {buckets.map((count, i) => (
        <span key={i} className="adm-hist-col" title={`${i * 10}–${i * 10 + 10}%: ${count}`}
          style={{ height: `${Math.max(4, (count / max) * 100)}%`, animationDelay: `${i * 45}ms` }}>
          <b>{count > 0 ? count : ''}</b>
        </span>
      ))}
      <span className="adm-hist-axis">0%</span>
      <span className="adm-hist-axis end">100%</span>
    </div>
  )
}

/* ---------- headline stat ---------- */

export function StatCard({ index, label, countTo, format, sub }: {
  index: number; label: string; countTo: number; format: (n: number) => string; sub: string
}) {
  const n = useCountUp(countTo)
  return (
    <article className="adm-stat" style={{ '--i': index } as CSSProperties}>
      <span className="g-label">{label}</span>
      <p className="g-value">{format(n)}</p>
      <span className="g-sub">{sub}</span>
    </article>
  )
}
