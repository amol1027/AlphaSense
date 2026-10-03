import React, { lazy, useEffect, useRef, useState, Suspense } from 'react'
import './homepage.css'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { ArrowDown, TrendingUp, Zap, Shield, Activity, ChevronRight, ChevronDown, Database, Newspaper, BrainCircuit, ChartNoAxesCombined, ArrowUpRight, CircleCheck, Gauge } from 'lucide-react'

gsap.registerPlugin(ScrollTrigger)

// 3D scenes live in ./scenes so three.js ships in its own async chunk and the
// article content paints first. The closing canvas additionally waits until
// it scrolls near the viewport (see closing3d below).
const HeroCanvas = lazy(() => import('./scenes').then(module => ({ default: module.HeroCanvas })))
const ClosingCanvas = lazy(() => import('./scenes').then(module => ({ default: module.ClosingCanvas })))

function useInView<T extends HTMLElement>(rootMargin = '400px') {
  const ref = useRef<T>(null)
  const [inView, setInView] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    if (typeof IntersectionObserver === 'undefined') { setInView(true); return }
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) {
        setInView(true)
        observer.disconnect()
      }
    }, { rootMargin })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])
  return { ref, inView }
}

// ─── Live feed strip ──────────────────────────────────────────────────────────
// Silent teaser for dashboard liveness. Renders nothing when the market
// service is unreachable so local/dev builds without a backend stay clean.
const MARKET_API = (import.meta.env.VITE_MARKET_API_BASE_URL || '/market-api').replace(/\/$/, '')

type FeedState = { state: string; latest_bar_timestamp: string | null; age_minutes: number | null }

function LiveStrip({ onNavigate, user }: HomePageProps) {
  const [feed, setFeed] = useState<FeedState | null>(null)
  useEffect(() => {
    let live = true
    const load = () => {
      fetch(`${MARKET_API}/dashboard`)
        .then(response => (response.ok ? response.json() : null))
        .then(data => { if (live && data?.feed?.latest_bar_timestamp) setFeed(data.feed) })
        .catch(() => { /* backend down: strip stays hidden */ })
    }
    load()
    const timer = window.setInterval(load, 60_000)
    return () => { live = false; window.clearInterval(timer) }
  }, [])
  if (!feed?.latest_bar_timestamp) return null
  const go = (e: React.MouseEvent<HTMLAnchorElement>, path: string) => { e.preventDefault(); onNavigate(path) }
  const time = new Date(feed.latest_bar_timestamp).toLocaleString('en-IN', {
    timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', hourCycle: 'h23',
  })
  const age = feed.age_minutes == null ? '' : feed.age_minutes < 60
    ? `${Math.round(feed.age_minutes)} MIN AGO`
    : `${(feed.age_minutes / 60).toFixed(1)} HOURS AGO`
  const label = feed.state === 'fresh' ? 'MARKET DATA IS FRESH'
    : feed.state === 'delayed' ? 'MARKET DATA IS DELAYED'
    : feed.state === 'stale' ? 'MARKET DATA IS STALE' : 'MARKET DATA'
  const dot = feed.state === 'fresh' ? 'bg-teal' : feed.state === 'delayed' ? 'bg-accent' : 'bg-muted'
  const target = user ? '/markets' : '/login?next=%2Fmarkets'
  return <div className="border-y border-hairline" aria-live="polite">
    <div className="mx-auto flex w-full max-w-[1500px] flex-wrap items-center justify-between gap-x-6 gap-y-2 px-6 py-3 md:px-24">
      <p className="flex items-center gap-2.5 font-mono text-[9px] tracking-[0.09em] text-muted">
        <span aria-hidden="true" className={`inline-block h-[7px] w-[7px] animate-pulse rounded-full ${dot}`} />
        {label} · LAST COMPLETED BAR {time} IST{age ? ` · ${age}` : ''}
      </p>
      <a
        className="font-mono text-[9px] tracking-[0.09em] text-ink underline decoration-accent underline-offset-4 hover:text-accent"
        href={target}
        onClick={e => go(e, target)}
      >
        {user ? 'OPEN DASHBOARD →' : 'SIGN IN TO VIEW LIVE SIGNALS →'}
      </a>
    </div>
  </div>
}

// ─── Stat Card ────────────────────────────────────────────────────────────────

function StatCard({ value, label, sub, accent, delay, countTo, prefix = '', suffix = '', decimals = 0 }: {
  value: string; label: string; sub: string; accent?: boolean; delay: number
  countTo?: number; prefix?: string; suffix?: string; decimals?: number
}) {
  const ref = useRef<HTMLDivElement>(null)
  const valueRef = useRef<HTMLSpanElement>(null)
  useEffect(() => {
    if (!ref.current) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const ctx = gsap.context(() => {
      gsap.fromTo(ref.current,
        { y: 32, opacity: 0, clipPath: 'inset(0 0 100% 0)' },
        { y: 0, opacity: 1, clipPath: 'inset(0 0 0% 0)', duration: 0.72, delay, ease: 'power3.out',
          scrollTrigger: { trigger: ref.current, start: 'top 88%', once: true } }
      )
      // Count-up: the rendered `value` is the final state (no-JS / reduced
      // motion safe); JS tweens the text from zero when scrolled into view.
      if (countTo != null && valueRef.current) {
        const target = valueRef.current
        const state = { v: 0 }
        gsap.to(state, {
          v: countTo, duration: 1.5, delay, ease: 'expo.out',
          scrollTrigger: { trigger: ref.current, start: 'top 88%', once: true },
          onUpdate: () => { target.textContent = `${prefix}${state.v.toFixed(decimals)}${suffix}` },
        })
      }
    }, ref)
    return () => ctx.revert()
  }, [delay, countTo, prefix, suffix, decimals])
  return (
    <div ref={ref} className={`hp-stat-card${accent ? ' hp-stat-card--accent' : ''}`}>
      <span ref={valueRef} className="hp-stat-value">{value}</span>
      <span className="hp-stat-label">{label}</span>
      <span className="hp-stat-sub">{sub}</span>
    </div>
  )
}

// ─── Ticker tape ──────────────────────────────────────────────────────────────

const TICKER_DATA = [
  { sym: 'HDFCBANK', val: '+13.5pp', sign: 1 },
  { sym: 'ICICIBANK', val: '+10.7pp', sign: 1 },
  { sym: 'RELIANCE', val: '+14.3pp', sign: 1 },
  { sym: 'INFY', val: '+14.1pp', sign: 1 },
  { sym: 'TCS', val: '+14.8pp', sign: 1 },
  { sym: 'NSE · RANGE', val: '130/130', sign: 1 },
  { sym: 'MEDIAN LIFT', val: '+13.9pp', sign: 1 },
]

function TickerTape() {
  return (
    <div className="hp-ticker-wrap">
      <span className="hp-sr-only">Locked research lifts by asset: HDFCBANK +13.5pp, ICICIBANK +10.7pp, RELIANCE +14.3pp, INFY +14.1pp, TCS +14.8pp. Median lift +13.9pp across 130 of 130 folds.</span>
      <div className="hp-ticker-inner" aria-hidden="true">
        {[...TICKER_DATA, ...TICKER_DATA, ...TICKER_DATA].map((item, i) => (
          <span key={i} className="hp-ticker-item">
            <span className="hp-ticker-sym">{item.sym}</span>
            <span className="hp-ticker-val">{item.val}</span>
            <span className="hp-ticker-sep">·</span>
          </span>
        ))}
      </div>
    </div>
  )
}

// ─── Feature Row ──────────────────────────────────────────────────────────────

function FeatureRow({ icon: Icon, title, body, index }: {
  icon: React.ComponentType<{ size?: number }>; title: string; body: string; index: number
}) {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!ref.current) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const ctx = gsap.context(() => {
      gsap.fromTo(ref.current,
        { x: -24, opacity: 0 },
        { x: 0, opacity: 1, duration: 0.64, delay: index * 0.11, ease: 'power2.out',
          scrollTrigger: { trigger: ref.current, start: 'top 85%', once: true } }
      )
    }, ref)
    return () => ctx.revert()
  }, [index])
  return (
    <div ref={ref} className="hp-feature-row">
      <div className="hp-feature-icon-wrap"><Icon size={18} /></div>
      <div>
        <p className="hp-feature-title">{title}</p>
        <p className="hp-feature-body">{body}</p>
      </div>
    </div>
  )
}

// ─── Fold Matrix (animated) ───────────────────────────────────────────────────

const PILOT_ASSETS = [
  { symbol: 'HDFCBANK', lift: '+13.5pp' },
  { symbol: 'ICICIBANK', lift: '+10.7pp' },
  { symbol: 'INFY', lift: '+14.1pp' },
  { symbol: 'RELIANCE', lift: '+14.3pp' },
  { symbol: 'TCS', lift: '+14.8pp' },
]

function FoldMatrix() {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!ref.current) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const ctx = gsap.context(() => {
      const cells = gsap.utils.toArray<HTMLElement>('.hp-fold-cell', ref.current)
      gsap.fromTo(cells,
        { scaleY: 0.08, opacity: 0.15 },
        { scaleY: 1, opacity: 1, duration: 0.3, stagger: 0.008, ease: 'power2.out', transformOrigin: 'bottom center',
          scrollTrigger: { trigger: ref.current, start: 'top 82%', once: true } }
      )
    }, ref)
    return () => ctx.revert()
  }, [])
  return (
    <div ref={ref} className="hp-fold-matrix">
      {PILOT_ASSETS.map(asset => (
        <div key={asset.symbol} className="hp-fold-lane">
          <span className="hp-fold-sym">{asset.symbol}</span>
          <span className="hp-fold-cells">
            {Array.from({ length: 26 }, (_, i) => <i key={i} className="hp-fold-cell" />)}
          </span>
          <span className="hp-fold-lift">{asset.lift}</span>
        </div>
      ))}
    </div>
  )
}

// These are the locked-holdout balanced accuracies documented in phase_4_pilot.md.
// They are deliberately not presented as live prices or trading recommendations.
const SNAPSHOT_ASSETS = [
  { symbol: 'HDFCBANK', locked: '0.682', lift: '+13.5pp', bars: [22, 35, 27, 48, 42, 61, 57, 74, 68, 82] },
  { symbol: 'ICICIBANK', locked: '0.541', lift: '+10.7pp', bars: [36, 25, 40, 32, 49, 47, 56, 44, 62, 58] },
  { symbol: 'INFY', locked: '0.673', lift: '+14.1pp', bars: [18, 30, 24, 44, 39, 52, 49, 66, 61, 78] },
  { symbol: 'RELIANCE', locked: '0.539', lift: '+14.3pp', bars: [44, 37, 50, 34, 55, 46, 61, 53, 58, 49] },
  { symbol: 'TCS', locked: '0.660', lift: '+14.8pp', bars: [26, 39, 31, 47, 43, 59, 54, 69, 63, 75] },
]

function MiniSignal({ bars }: { bars: number[] }) {
  const points = bars.map((value, index) => `${index * 11.1},${96 - value}`).join(' ')
  return <svg className="hp-mini-signal" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
    <polyline points={points} pathLength={100} />
  </svg>
}

function MarketSnapshot() {
  const tableRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!tableRef.current) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const ctx = gsap.context(() => {
      const trigger = { trigger: tableRef.current, start: 'top 80%', once: true }
      gsap.fromTo('.hp-asset-row',
        { y: 26, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.7, stagger: 0.09, ease: 'power3.out', scrollTrigger: trigger }
      )
      // Sparklines draw themselves left-to-right as their row arrives.
      gsap.utils.toArray<SVGPolylineElement>('.hp-mini-signal polyline').forEach((line, i) => {
        gsap.fromTo(line,
          { strokeDashoffset: 100 },
          { strokeDashoffset: 0, duration: 1.1, delay: 0.25 + i * 0.09, ease: 'power2.inOut', scrollTrigger: trigger }
        )
      })
    }, tableRef)
    return () => ctx.revert()
  }, [])
  return <section className="hp-snapshot" id="signals" aria-labelledby="hp-snapshot-heading">
    <div className="hp-section-intro" data-reveal>
      <div><p className="hp-section-eyebrow">MARKET SNAPSHOT / RESEARCH SET</p><h2 id="hp-snapshot-heading" className="hp-section-h2">Five assets. One<br />measurable question.</h2></div>
      <p>Historical 15-minute NSE observations are transformed into a next-hour range-regime target. The instrument below reports locked research evaluation, not prices or investable signals.</p>
    </div>
    <div className="hp-snapshot-table" role="table" aria-label="Locked evaluation by asset" ref={tableRef}>
      <div className="hp-snapshot-head" role="row"><span>ASSET</span><span>RANGE REGIME / 1H</span><span>LOCKED BA</span><span>MEDIAN LIFT</span></div>
      {SNAPSHOT_ASSETS.map((asset, index) => <article className="hp-asset-row" role="row" key={asset.symbol} style={{ '--row-delay': `${index * 70}ms` } as React.CSSProperties}>
        <strong role="cell">{asset.symbol}</strong><div className="hp-signal-track" role="cell"><MiniSignal bars={asset.bars} /><span>EXPANDING WALK-FORWARD</span></div><span className="hp-locked-value" role="cell">{asset.locked}</span><span className="hp-lift-value" role="cell">{asset.lift}</span>
      </article>)}
    </div>
    <p className="hp-source-note">BA = balanced accuracy. Locked holdout scored once after model selection. Source: Phase 4 pilot, 2024–2026 research data.</p>
  </section>
}

function IntelligenceMap() {
  const [active, setActive] = useState<'market' | 'news' | 'model'>('market')
  const detail = {
    market: ['MARKET FEATURES', 'Current-bar range, return structure, and lagged one-hour measures form the minimal feature set that held up across assets.'],
    news: ['NEWS / SENTIMENT', 'News and attention features were evaluated as additions. At this coverage and representation, neither produced stable incremental lift.'],
    model: ['EVALUATION GATE', 'Expanding walk-forward folds and a locked holdout decide whether a relationship earns a research finding.'],
  }[active]
  const nodeButton = (id: 'market' | 'news' | 'model', label: string, Icon: React.ComponentType<{ size?: number }>) => (
    <button key={id} type="button" onClick={() => setActive(id)} className={`hp-network-node ${active === id ? 'is-active' : ''}`} aria-pressed={active === id}><Icon size={18} /><span>{label}</span></button>
  )
  return <section className="hp-intelligence" aria-labelledby="hp-intelligence-heading">
    <div className="hp-intelligence-copy"><p className="hp-section-eyebrow">RESEARCH INTELLIGENCE</p><h2 id="hp-intelligence-heading" className="hp-section-h2">Evidence is a<br /><em>relationship.</em></h2><p>Signals are not decoration. Each source is tested against the same temporal cutoff, target definition, and validation rule before it is allowed into the record.</p></div>
    <div className="hp-network-shell">
      <div className="hp-network" aria-label="Interactive diagram of research signal inputs">
        <div className="hp-flow-inputs">
          {nodeButton('market', 'MARKET', ChartNoAxesCombined)}
          {nodeButton('news', 'NEWS', Newspaper)}
        </div>
        <svg className="hp-flow-wires" viewBox="0 0 56 200" preserveAspectRatio="none" aria-hidden="true" focusable="false">
          <path d="M2 44 C 30 44, 26 100, 54 100" className={active === 'market' ? 'is-live' : ''} />
          <path d="M2 156 C 30 156, 26 100, 54 100" className={active === 'news' ? 'is-live' : ''} />
        </svg>
        {nodeButton('model', 'MODEL', BrainCircuit)}
        <span className="hp-flow-link" aria-hidden="true" />
        <div className="hp-network-outcome"><Gauge size={18} /><span>RANGE<br />REGIME</span></div>
      </div>
      <div className="hp-network-readout" aria-live="polite"><span>INSPECT / {active.toUpperCase()}</span><h3>{detail[0]}</h3><p>{detail[1]}</p></div>
    </div>
  </section>
}

function Findings() {
  const findings: Array<[string, string, string, string]> = [
    ['01', 'Range outperformed direction.', 'Next-hour direction did not demonstrate an edge across 130 pooled market-only folds. Range regime did.', 'RANGE HELD'],
    ['02', 'The shortest useful horizon held.', '30 minutes and 1 hour remained strong; signal weakened steadily at 2 and 4 hours. The one-hour specification remains frozen.', '1H FROZEN'],
    ['03', 'More data is not automatically more signal.', 'News sentiment and Google Trends were tested on top of the baseline but did not add stable incremental value.', 'NO LIFT'],
  ]
  return <section className="hp-findings" id="findings" aria-labelledby="hp-findings-heading"><div className="hp-findings-top" data-reveal><div className="hp-findings-toprow"><p className="hp-section-eyebrow">RESEARCH FINDINGS / PHASE 4</p><span>EXPLORATORY RESEARCH · NOT INVESTMENT ADVICE</span></div><h2 id="hp-findings-heading" className="hp-section-h2">What the data<br />actually permits.</h2></div><div className="hp-findings-list">{findings.map(([number, title, body, tag]) => <article key={number} className="hp-finding"><span>{number}</span><div><h3>{title}</h3><p>{body}</p></div><em className="hp-finding-tag">{tag}</em><CircleCheck size={18} aria-hidden="true" /></article>)}</div></section>
}

function Methodology() {
  const steps: Array<[string, string, string, string, boolean?]> = [
    ['01', 'Market data', '15-minute OHLCV research set', 'RESEARCH SET'],
    ['02', 'Feature engineering', 'Lagged, point-in-time measures', 'POINT-IN-TIME'],
    ['03', 'News & sentiment', 'Tested as incremental inputs', 'NO STABLE LIFT', true],
    ['04', 'Model evaluation', 'Expanding walk-forward + locked set', '130 FOLDS'],
    ['05', 'Research finding', 'Only stable results are promoted', 'FROZEN'],
  ]
  return <section className="hp-methodology" id="methodology" aria-labelledby="hp-method-heading"><div className="hp-method-head" data-reveal><p className="hp-section-eyebrow">METHODOLOGY / THE RESEARCH GATE</p><h2 id="hp-method-heading" className="hp-section-h2">Nothing crosses the cutoff by accident.</h2><p>Every candidate signal is prepared using only information available at prediction time. Future observations build the target, never the features.</p><div className="hp-method-foot"><span>05 STEPS · 01 GATE</span><a href="#cutoff">See the cutoff in action ↓</a></div></div><ol className="hp-method-steps">{steps.map(([number, title, body, tag, warn]) => <li key={number}><span>{number}</span><Database size={16} /><div><strong>{title}</strong><small>{body}</small></div><em className={`hp-method-tag${warn ? ' hp-method-tag--warn' : ''}`}>{tag}</em></li>)}</ol></section>
}

const FAQ_ITEMS: Array<[string, string]> = [
  ['Why range, not direction?', 'Next-hour direction showed no stable edge across 130 pooled market-only folds. Classifying whether the next hour’s range exceeds its training median did — consistently, on fully held-out windows.'],
  ['How do you prevent data leakage?', 'Every feature must end at or before prediction time t. Only completed 15-minute bars enter the inputs; the next hour’s high–low window builds the target label and is never used as an input.'],
  ['Which horizon actually works?', '30 minutes and 1 hour remained strong. The signal weakened steadily at 2 and 4 hours, so the one-hour specification is frozen and evaluated out-of-sample.'],
  ['Did news and sentiment help?', 'Tested as incremental inputs on top of the market baseline. At this coverage and representation, neither news sentiment nor search trends added stable lift — so they stay out of the finding.'],
  ['Is this investment advice?', 'No. AlphaSense is an exploratory research prototype on delayed 15-minute NSE data. It studies what past data can reveal, not what you should trade.'],
]

function Faq() {
  const [open, setOpen] = useState<number | null>(0)
  return <section id="faq" aria-labelledby="hp-faq-heading" className="mx-auto w-full max-w-[1500px] border-t border-hairline px-6 py-10 md:px-24 md:py-16">
    <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
      <div>
        <p className="mb-3 font-mono text-[9px] font-medium tracking-[0.14em] text-accent">FAQ / STRAIGHT ANSWERS</p>
        <h2 id="hp-faq-heading" className="font-serif text-4xl font-light tracking-tight text-ink md:text-5xl">Asked, answered,<br />no hedging.</h2>
      </div>
      <span className="font-mono text-[8px] tracking-[0.09em] text-muted">5 QUESTIONS · 30-SECOND READS</span>
    </div>
    <div className="border-t border-hairline">
      {FAQ_ITEMS.map(([q, a], i) => {
        const isOpen = open === i
        return <div key={q} className="border-b border-hairline">
          <button
            type="button"
            onClick={() => setOpen(isOpen ? null : i)}
            aria-expanded={isOpen}
            className="flex w-full cursor-pointer items-center justify-between gap-4 py-5 text-left"
          >
            <span className="flex items-baseline gap-4">
              <span className="font-mono text-[10px] text-accent">0{i + 1}</span>
              <span className="font-serif text-xl text-ink md:text-2xl">{q}</span>
            </span>
            <ChevronDown size={18} className={`shrink-0 text-teal transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`} />
          </button>
          <div className={`grid transition-all duration-300 ease-out ${isOpen ? 'grid-rows-[1fr] pb-5 opacity-100' : 'grid-rows-[0fr] opacity-0'}`}>
            <p className="max-w-2xl overflow-hidden text-sm leading-relaxed text-muted">{a}</p>
          </div>
        </div>
      })}
    </div>
  </section>
}

// ─── Scroll Progress ──────────────────────────────────────────────────────────

function useScrollProgress() {
  const [prog, setProg] = useState(0)
  useEffect(() => {
    const update = () => {
      const max = document.documentElement.scrollHeight - window.innerHeight
      setProg(max > 0 ? window.scrollY / max : 0)
    }
    window.addEventListener('scroll', update, { passive: true })
    update()
    return () => window.removeEventListener('scroll', update)
  }, [])
  return prog
}

// ─── Glitch text ──────────────────────────────────────────────────────────────

function GlitchText({ text }: { text: string }) {
  return (
    <span className="hp-glitch" data-text={text} aria-label={text}>
      {text}
    </span>
  )
}

// ─── Magnetic link ────────────────────────────────────────────────────────
// Primary/ghost CTAs drift a few px toward the cursor and snap back on leave.
// Purely decorative: keyboard and reduced-motion users get the static button.

function MagneticLink({ className, href, onClick, children }: {
  className: string; href: string; onClick?: (e: React.MouseEvent<HTMLAnchorElement>) => void; children: React.ReactNode
}) {
  const ref = useRef<HTMLAnchorElement>(null)
  const move = (event: React.MouseEvent) => {
    const el = ref.current
    if (!el || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const rect = el.getBoundingClientRect()
    const x = (event.clientX - rect.left - rect.width / 2) / rect.width
    const y = (event.clientY - rect.top - rect.height / 2) / rect.height
    el.style.transform = `translate(${(x * 8).toFixed(1)}px, ${(y * 6).toFixed(1)}px)`
  }
  const reset = () => { if (ref.current) ref.current.style.transform = '' }
  return <a ref={ref} className={`${className} hp-magnetic`} href={href} onClick={onClick} onMouseMove={move} onMouseLeave={reset}>{children}</a>
}

// ─── Main Export ──────────────────────────────────────────────────────────────

interface HomePageProps {
  onNavigate: (path: string) => void
  user?: { displayName: string; email: string } | null
}

export function HomePage({ onNavigate, user }: HomePageProps) {
  const heroRef = useRef<HTMLElement>(null)
  const canvasWrapRef = useRef<HTMLDivElement>(null)
  const headlineRef = useRef<HTMLHeadingElement>(null)
  const scrollProg = useScrollProgress()
  const pageRef = useRef<HTMLDivElement>(null)
  const [reducedMotion, setReducedMotion] = useState(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches)

  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)')
    const update = () => setReducedMotion(media.matches)
    media.addEventListener('change', update)
    return () => media.removeEventListener('change', update)
  }, [])

  // Entrance animations
  useEffect(() => {
    const settleHeadline = () => {
      // The overflow mask is only needed while lines wipe in; afterwards it
      // would clip Newsreader's descenders, so release it once settled.
      headlineRef.current?.querySelectorAll<HTMLElement>('.hp-headline-line').forEach(el => {
        el.style.overflow = 'visible'
      })
    }
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      settleHeadline()
      return
    }
    const tl = gsap.timeline({ defaults: { ease: 'power3.out' } })
    tl.fromTo('.hp-eyebrow', { clipPath: 'inset(0 100% 0 0)', x: -16 }, { clipPath: 'inset(0 0% 0 0)', x: 0, duration: 0.58 }, 0.1)
      .fromTo('.hp-headline-line', { clipPath: 'inset(0 0 100% 0)' }, { clipPath: 'inset(0 0 0% 0)', duration: 0.72, stagger: 0.14 }, 0.28)
      .fromTo('.hp-sub', { opacity: 0, y: 18 }, { opacity: 1, y: 0, duration: 0.6 }, 0.72)
      .fromTo('.hp-cta-group', { opacity: 0, y: 14 }, { opacity: 1, y: 0, duration: 0.5 }, 0.92)
      .fromTo(canvasWrapRef.current, { opacity: 0, scale: 0.94 }, { opacity: 1, scale: 1, duration: 1.0, ease: 'power2.out' }, 0.2)
      .add(settleHeadline, 1.35)
  }, [])

  // Scroll systems: hero canvas parallax + generic reveals for static blocks.
  // Blocks already animated elsewhere (stats, features, fold matrix, snapshot
  // rows, LeakageExplorer) must NOT carry data-reveal.
  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const ctx = gsap.context(() => {
      if (canvasWrapRef.current && heroRef.current) {
        gsap.to(canvasWrapRef.current, {
          yPercent: 10, ease: 'none',
          scrollTrigger: { trigger: heroRef.current, start: 'top top', end: 'bottom top', scrub: true },
        })
      }
      gsap.utils.toArray<HTMLElement>('[data-reveal]').forEach(el => {
        gsap.fromTo(el,
          { y: 30, opacity: 0 },
          { y: 0, opacity: 1, duration: 0.85, ease: 'power3.out',
            delay: Number(el.dataset.revealDelay ?? 0),
            scrollTrigger: { trigger: el, start: 'top 87%', once: true } }
        )
      })
    }, pageRef)
    return () => ctx.revert()
  }, [])

  const go = (e: React.MouseEvent<HTMLAnchorElement>, path: string) => { e.preventDefault(); onNavigate(path) }

  return (
    <div className="hp-root" ref={pageRef}>
      {/* Reading progress bar */}
      <div className="hp-progress" style={{ transform: `scaleX(${scrollProg})` }} aria-hidden="true" />

      {/* ── HERO ────────────────────────────────────────────────── */}
      <section ref={heroRef} className="hp-hero" id="top" aria-labelledby="hp-heading">

        {/* 3D Canvas (async chunk; content paints first) */}
        <div ref={canvasWrapRef} className="hp-canvas-wrap" aria-hidden="true">
          <Suspense fallback={null}>
            <HeroCanvas reducedMotion={reducedMotion} />
          </Suspense>
          <div className="hp-scene-frame">
            <div className="hp-scene-topline"><span><i /> MARKET STRUCTURE / 3D STUDY</span><span>INDIA · 15 MIN</span></div>
            <div className="hp-scene-axis hp-scene-axis--left"><span>RANGE</span><i /><i /><i /><i /><i /><span>TIME →</span></div>
            <div className="hp-scene-callout"><span>MODEL TARGET</span><strong>VOLATILITY</strong><small>Next-hour range regime</small></div>
            <div className="hp-scene-bottomline"><span>DRAG TO ORBIT · HOVER TO INSPECT · CLICK FOR BURST</span><span>NOT LIVE PRICES</span></div>
          </div>
          <div className="hp-canvas-vignette" />
        </div>

        {/* Hero copy */}
        <div className="hp-hero-content">
          <div className="hp-eyebrow">
            <span className="hp-eyebrow-dot" />
            <span>RESEARCH NOTEBOOK</span>
            <span className="hp-eyebrow-sep">·</span>
            <span>VOL. 01 / 2026</span>
            <span className="hp-eyebrow-sep">·</span>
            <span>NSE · BSE</span>
          </div>

          <h1 id="hp-heading" className="hp-headline" ref={headlineRef}>
            <span className="hp-headline-line hp-headline-line--1">
              <GlitchText text="Direction" />
              <span className="hp-headline-muted"> was noise.</span>
            </span>
            <span className="hp-headline-line hp-headline-line--2">
              Volatility<span className="hp-headline-accent"> held</span>
            </span>
            <span className="hp-headline-line hp-headline-line--3 hp-headline-muted">
              a signal.
            </span>
          </h1>

          <p className="hp-sub">
            We study what Indian market data can reveal about the next hour of trading.
            <br />
            Range regime outperforms direction — consistently, across 130 held-out folds.
          </p>

          <div className="hp-cta-group">
            {user ? (
              <MagneticLink className="hp-cta-primary" href="/markets" onClick={e => go(e, '/markets')}>
                Open dashboard
                <ChevronRight size={16} />
              </MagneticLink>
            ) : (
              <MagneticLink className="hp-cta-primary" href="/login" onClick={e => go(e, '/login?next=%2Fmarkets')}>
                Sign in to view live signals
                <ChevronRight size={16} />
              </MagneticLink>
            )}
            <MagneticLink className="hp-cta-ghost" href="#methodology" onClick={e => go(e, '/#methodology')}>
              View Methodology
              <Activity size={14} />
            </MagneticLink>
          </div>
        </div>

        {/* Scroll cue */}
        <div className="hp-scroll-cue" aria-hidden="true">
          <div className="hp-scroll-line" />
          <ArrowDown size={13} />
        </div>
      </section>

      {/* ── TICKER ──────────────────────────────────────────────── */}
      <TickerTape />

      {/* ── LIVE FEED STRIP ─────────────────────────────────────── */}
      <LiveStrip onNavigate={onNavigate} user={user} />

      {/* ── STATS ROW ───────────────────────────────────────────── */}
      <section className="hp-stats-section" aria-label="Key results">
        <StatCard value="130/130" label="Walk-Forward Folds" sub="Beat majority baseline" accent delay={0} countTo={130} suffix="/130" />
        <StatCard value="+13.9pp" label="Median Accuracy Lift" sub="Over naive baseline" delay={0.12} countTo={13.9} prefix="+" suffix="pp" decimals={1} />
        <StatCard value="5" label="NSE Assets" sub="HDFC · ICICI · INFY · RIL · TCS" delay={0.22} countTo={5} />
        <StatCard value="1H" label="Prediction Horizon" sub="Frozen, expanding-window" delay={0.32} countTo={1} suffix="H" />
      </section>

      {/* ── EVIDENCE PANEL ──────────────────────────────────────── */}
      <section className="hp-evidence-section" aria-labelledby="hp-evidence-heading">
        <div className="hp-evidence-inner">
          <div className="hp-evidence-left">
            <p className="hp-section-eyebrow">THE SIGNAL THAT SURVIVED</p>
            <h2 id="hp-evidence-heading" className="hp-section-h2">130 folds.<br />Zero failures.</h2>
            <p className="hp-evidence-deck">
              Every single held-out fold beat the majority baseline — across five assets and 26 monthly test windows each. The range regime signal is not a fluke.
            </p>
            <div className="hp-features">
              <FeatureRow icon={Shield} title="No Data Leakage" body="Strict temporal cutoff: only data available at prediction time t enters any feature." index={0} />
              <FeatureRow icon={Zap} title="Frozen 1H Model" body="Model weights locked after training. Performance measured on fully out-of-sample folds." index={1} />
              <FeatureRow icon={TrendingUp} title="Range, Not Direction" body="The target is whether the next hour's range exceeds its training median — not price direction." index={2} />
            </div>
          </div>
          <div className="hp-evidence-right">
            <div className="hp-fold-card">
              <div className="hp-fold-card-top">
                <span>WALK-FORWARD / EXPANDING WINDOW</span>
                <span className="hp-fold-card-badge">130/130 ✓</span>
              </div>
              <div className="hp-fold-big">
                <span className="hp-fold-num">130</span>
                <span className="hp-fold-denom">/130</span>
                <span className="hp-fold-label">folds beat<br />majority baseline</span>
              </div>
              <FoldMatrix />
              <div className="hp-fold-axis">
                <span>26 MONTHLY TEST WINDOWS / ASSET</span>
                <span>+13.9pp MEDIAN LIFT</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── BRIDGE / RESEARCH Q ─────────────────────────────────── */}
      <section className="hp-bridge" id="question">
        <div className="hp-bridge-inner" data-reveal>
          <span className="hp-bridge-kicker">THE RESEARCH QUESTION</span>
          <p className="hp-bridge-q">
            Can the next hour's trading range be classified as{' '}
            <em>above or below</em> its training-period median?
          </p>
          <span className="hp-bridge-note">PHASE 4 · PILOT · 5 STOCKS · 130 WALK-FORWARD FOLDS</span>
        </div>
        <a className="hp-bridge-cta" data-reveal data-reveal-delay="0.12" href="#cutoff" onClick={e => go(e, '/#cutoff')}>
          Explore the Method <ArrowDown size={14} />
        </a>
      </section>

      <MarketSnapshot />
      <IntelligenceMap />
      <Findings />
      <Methodology />
      <Faq />
    </div>
  )
}

// ─── Closing / About (directly above the footer) ─────────────────────────
// Rendered after the cutoff explorer in main.tsx. Self-contained: own
// reduced-motion state, reveal animations, and lazy 3D — wrapped in
// .hp-root so theme tokens and type apply outside the homepage flow.
export function ClosingSection({ onNavigate, user }: HomePageProps) {
  const rootRef = useRef<HTMLDivElement>(null)
  const closing3d = useInView<HTMLDivElement>()
  const [reducedMotion, setReducedMotion] = useState(() => window.matchMedia('(prefers-reduced-motion: reduce)').matches)

  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)')
    const update = () => setReducedMotion(media.matches)
    media.addEventListener('change', update)
    return () => media.removeEventListener('change', update)
  }, [])

  useEffect(() => {
    if (!rootRef.current) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const ctx = gsap.context(() => {
      gsap.utils.toArray<HTMLElement>('[data-reveal]').forEach(el => {
        gsap.fromTo(el,
          { y: 30, opacity: 0 },
          { y: 0, opacity: 1, duration: 0.85, ease: 'power3.out',
            delay: Number(el.dataset.revealDelay ?? 0),
            scrollTrigger: { trigger: el, start: 'top 87%', once: true } }
        )
      })
    }, rootRef)
    return () => ctx.revert()
  }, [])

  const go = (e: React.MouseEvent<HTMLAnchorElement>, path: string) => { e.preventDefault(); onNavigate(path) }

  return (
    <div className="hp-root" ref={rootRef}>
      <section className="hp-closing" id="about" aria-labelledby="hp-closing-heading">
        <div className="hp-closing-grid" aria-hidden="true" />
        <div ref={closing3d.ref} className="hp-closing-3d" aria-hidden="true">
          {closing3d.inView && <Suspense fallback={null}>
            <ClosingCanvas reducedMotion={reducedMotion} />
          </Suspense>}
        </div>
        <p className="hp-section-eyebrow" data-reveal>ALPHASENSE / ONGOING RESEARCH</p>
        <h2 id="hp-closing-heading" data-reveal data-reveal-delay="0.08">Research with a<br /><em>record of restraint.</em></h2>
        <p data-reveal data-reveal-delay="0.16">Explore the full methodology, validation constraints, and live research workspace. The goal is not a louder market narrative—it is a more defensible one.</p>
        <div className="hp-closing-actions" data-reveal data-reveal-delay="0.24"><MagneticLink className="hp-cta-primary" href="#methodology">View research methodology <ArrowUpRight size={16} /></MagneticLink>{user ? <MagneticLink className="hp-cta-ghost" href="/markets" onClick={e => go(e, '/markets')}>Open dashboard <Activity size={14} /></MagneticLink> : <MagneticLink className="hp-cta-ghost" href="/login" onClick={e => go(e, '/login?next=%2Fmarkets')}>Sign in to view live signals <Activity size={14} /></MagneticLink>}</div>
        <dl className="relative z-[1] mt-10 flex max-w-xl flex-wrap gap-x-8 gap-y-4 border-t border-hairline pt-6">
          {[
            ['130/130', 'folds beat baseline'],
            ['5', 'NSE assets'],
            ['15-min', 'IST bars'],
            ['1H', 'frozen horizon'],
          ].map(([v, l]) => (
            <div key={l} className="flex items-baseline gap-2">
              <dt className="font-mono text-sm text-ink">{v}</dt>
              <dd className="m-0 font-mono text-[9px] uppercase tracking-[0.08em] text-muted">{l}</dd>
            </div>
          ))}
        </dl>
        <p className="relative z-[1] mt-4 max-w-xl font-mono text-[9px] leading-relaxed tracking-[0.04em] text-muted">EXPLORATORY RESEARCH · NOT INVESTMENT ADVICE · NEWS / TRENDS ADDED NO STABLE LIFT</p>
      </section>
    </div>
  )
}
