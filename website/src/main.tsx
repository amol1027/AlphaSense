import React, { Suspense, lazy, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { ArrowDown, Menu, Moon, Sun, X } from 'lucide-react'
import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { getCurrentUser, signOut, type AuthUser } from './auth-client'
import { SiteFooter } from './SiteFooter'
import { useTheme } from './theme'
import './styles.css'

// Route-split: three.js (HomePage) and the dashboard stay out of the initial
// chunk so /login and first paint stay light. Each suspends with a skeleton.
const HomePage = lazy(() => import('./HomePage').then(module => ({ default: module.HomePage })))
const ClosingSection = lazy(() => import('./HomePage').then(module => ({ default: module.ClosingSection })))
const MarketDashboard = lazy(() => import('./MarketDashboard').then(module => ({ default: module.MarketDashboard })))
const AuthPage = lazy(() => import('./AuthPage').then(module => ({ default: module.AuthPage })))
const ProfilePage = lazy(() => import('./ProfilePage').then(module => ({ default: module.ProfilePage })))
const AdminDashboard = lazy(() => import('./AdminDashboard').then(module => ({ default: module.AdminDashboard })))

function useIstClock() {
  const [now, setNow] = useState(new Date())
  useEffect(() => { const timer = window.setInterval(() => setNow(new Date()), 1000); return () => clearInterval(timer) }, [])
  const parts = new Intl.DateTimeFormat('en-IN', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23', weekday: 'short' }).formatToParts(now)
  const get = (type: string) => parts.find(part => part.type === type)?.value ?? ''
  const hour = Number(get('hour')), minute = Number(get('minute'))
  const weekday = get('weekday')
  const open = !['Sat', 'Sun'].includes(weekday) && (hour * 60 + minute >= 555) && (hour * 60 + minute < 930)
  return { time: `${get('hour')}:${get('minute')}:${get('second')}`, open }
}

function Header({ user, marketPage, adminPage, onNavigate, onSignOut }: { user: AuthUser | null; marketPage?: boolean; adminPage?: boolean; onNavigate: (path: string) => void; onSignOut: () => void }) {
  const { time, open } = useIstClock()
  const [dark, setDark] = useTheme()
  const [menuOpen, setMenuOpen] = useState(false)
  const go = (event: React.MouseEvent<HTMLAnchorElement>, path: string) => { event.preventDefault(); setMenuOpen(false); onNavigate(path) }
  useEffect(() => {
    if (!menuOpen) return
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') setMenuOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [menuOpen])
  // Auth links remember where the user was so sign-in can send them back.
  // An existing ?next= wins (e.g. switching login ↔ register); otherwise the
  // current page is captured, except auth pages themselves (avoids loops).
  const authLink = (mode: 'login' | 'register') => {
    const params = new URLSearchParams(window.location.search)
    const fromParam = params.get('next')
    const here = normalizePathname(window.location.pathname)
    const from = fromParam ?? ((here === '/login' || here === '/register') ? '/' : here)
    const next = safeNext(from)
    return `/${mode}${next === '/' ? '' : `?next=${encodeURIComponent(next)}`}`
  }
  return <header className="topbar">
    <button className="menu-toggle" type="button" onClick={() => setMenuOpen(value => !value)} aria-label={menuOpen ? 'Close menu' : 'Open menu'} aria-expanded={menuOpen}>{menuOpen ? <X size={17} /> : <Menu size={17} />}</button>
    <a className="wordmark" href="/#top" onClick={event => go(event, '/#top')} aria-label="AlphaSense home"><span className="wordmark-mark" aria-hidden="true"><i>A</i><i>S</i></span><span>AlphaSense</span></a>
    <nav aria-label="Main navigation"><a href="/#findings" onClick={event => go(event, '/#findings')}>Research</a><a href="/#methodology" onClick={event => go(event, '/#methodology')}>Methodology</a><a href="/#about" onClick={event => go(event, '/#about')}>About</a>{user && <a href="/markets" onClick={event => go(event, '/markets')} aria-current={marketPage ? 'page' : undefined} className="nav-markets">Markets</a>}{user?.role === 'admin' && <a href="/admin" onClick={event => go(event, '/admin')} aria-current={adminPage ? 'page' : undefined} className="nav-markets">Admin</a>}</nav>
    <div className="header-status" title="India Standard Time · NSE 09:15–15:30"><span className="clock"><span className="clock-dot" />{time} IST</span><span className="session"><b className={open ? 'is-open' : ''}>{open ? 'OPEN' : 'CLOSED'}</b></span></div>
    <div className="account-links">{user ? <><a className="account-name" title={user.email} href="/profile" onClick={event => go(event, '/profile')}>{user.displayName}</a><button type="button" className="account-login" onClick={onSignOut}>Sign out</button></> : <><a className="account-login" href="/login" onClick={event => go(event, authLink('login'))}>Sign in</a><a className="account-register" href="/register" onClick={event => go(event, authLink('register'))}>Create account</a></>}</div>
    <button className="theme-toggle" type="button" onClick={() => setDark(value => !value)} aria-label={`Switch to ${dark ? 'light' : 'dark'} theme`} aria-pressed={dark}>{dark ? <Sun size={15} /> : <Moon size={15} />}</button>
    {menuOpen && <nav className="mobile-menu" aria-label="Mobile navigation">
      <a href="/#findings" onClick={event => go(event, '/#findings')}>Research</a>
      <a href="/#methodology" onClick={event => go(event, '/#methodology')}>Methodology</a>
      <a href="/#about" onClick={event => go(event, '/#about')}>About</a>
      {user && <a href="/markets" onClick={event => go(event, '/markets')}>Markets</a>}
      {user?.role === 'admin' && <a href="/admin" onClick={event => go(event, '/admin')}>Admin</a>}
      <span className="mobile-menu-rule" aria-hidden="true" />
      {user
        ? <><a className="mobile-menu-user" title={user.email} href="/profile" onClick={event => go(event, '/profile')}>{user.displayName}</a><button type="button" onClick={() => { setMenuOpen(false); onSignOut() }}>Sign out</button></>
        : <><a href="/login" onClick={event => go(event, authLink('login'))}>Sign in</a><a href="/register" onClick={event => go(event, authLink('register'))}>Create account</a></>}
    </nav>}
  </header>
}

type Datum = { name: string; time: string; state: 'pass'|'reject'; why: string; detail: string }
const examples: Datum[] = [
  { name: 'Last completed 15m close', time: '14:15 IST', state: 'pass', why: 'Available at or before t', detail: 'The candle has closed. Its close and range can be used to construct features at this timestamp.' },
  { name: 'Trailing 1h range', time: '13:15—14:15', state: 'pass', why: 'Window ends at t', detail: 'All four 15-minute bars are complete by the prediction time. No future bar enters the feature.' },
  { name: 'News published at 14:30', time: '14:30 IST', state: 'reject', why: 'Published after t', detail: 'This item arrived after the 14:15 prediction timestamp, so the model could not have known it.' },
  { name: 'Next hour high–low range', time: '14:15—15:15', state: 'reject', why: 'Outcome window is future', detail: 'This value defines the target label. It is used to score a prediction later, never as an input at t.' },
]

function LeakageExplorer() {
  const [selected, setSelected] = useState(0)
  const chosen = examples[selected]
  return <section id="cutoff" className="cutoff-section reveal-section" aria-labelledby="cutoff-heading">
    <div className="section-rail"><span>§4</span><span>METHOD / 01</span><span className="rail-rule" /></div>
    <div className="section-main">
      <div className="section-heading"><div><p className="eyebrow">TEMPORAL CONTRACT</p><h2 id="cutoff-heading">The cutoff is the method.</h2></div><p className="section-deck">At prediction time <em>t</em>, a feature can only know what has already happened. The boundary is strict.</p></div>
      <div className="explorer">
        <div className="timeline-panel">
          <div className="panel-topline"><span>FEATURE AVAILABILITY</span><span>14:15 IST <b>· t</b></span></div>
          <div className="timeline-track" aria-label="Feature availability timeline">
            <div className="timeline-known" /><div className="timeline-future" />
            {examples.map((item, index) => <button key={item.name} type="button" className={`timeline-point point-${index} ${selected === index ? 'selected' : ''} ${item.state}`} onMouseEnter={() => setSelected(index)} onFocus={() => setSelected(index)} onClick={() => setSelected(index)} aria-label={`${item.name}, ${item.state === 'pass' ? 'available' : 'rejected'}: ${item.why}`} aria-pressed={selected === index}><span className="point-marker" /><span className="point-name">{item.name}</span></button>)}
            <span className="timeline-cutoff"><i />t</span>
          </div>
          <div className="timeline-axis"><span>13:15</span><span>13:45</span><span>14:15</span><span>14:45</span><span>15:15</span></div>
          <div className="timeline-key"><span><i className="key-pass" />INFORMATION AVAILABLE</span><span><i className="key-reject" />FUTURE / REJECTED</span></div>
        </div>
        <article className={`inspection ${chosen.state}`} aria-live="polite" key={chosen.name}>
          <div className="inspection-top"><span className="mono">INSPECT / {String(selected + 1).padStart(2, '0')}</span><span className={`decision ${chosen.state}`}>{chosen.state === 'pass' ? 'PASSES CUTOFF' : 'REJECTED'}</span></div>
          <p className="inspection-time">{chosen.time}</p><h3>{chosen.name}</h3><p className="inspection-why">{chosen.why}</p><div className="inspection-rule" /><p className="inspection-detail">{chosen.detail}</p>
        </article>
      </div>
      <div className="method-footnote"><span>RULE 01</span><p>This same information cutoff applies to market features, news, and social signals. Future data builds the target only.</p><a href="#question" aria-label="Continue to the research question"><ArrowDown size={15} /></a></div>
    </div>
  </section>
}

function splitRoute(route: string) {
  let rest = route, hash = ''
  const hashIndex = rest.indexOf('#')
  if (hashIndex >= 0) {
    hash = rest.slice(hashIndex + 1)
    rest = rest.slice(0, hashIndex)
  }
  let search = ''
  const queryIndex = rest.indexOf('?')
  if (queryIndex >= 0) {
    search = rest.slice(queryIndex)
    rest = rest.slice(0, queryIndex)
  }
  return { pathname: rest || '/', search, hash }
}

function normalizePathname(pathname: string) {
  return pathname.length > 1 ? pathname.replace(/\/+$/, '') : pathname
}

/** Only internal app paths may be used as a post-auth destination. Query is preserved so deep links like /markets?stock=RELIANCE survive the login round-trip. */
function safeNext(raw: string | null): string {
  if (!raw) return '/'
  const { pathname: rawPath, search: rawSearch } = splitRoute(raw)
  const clean = normalizePathname(rawPath)
  if (!clean.startsWith('/') || clean.startsWith('//')) return '/'
  if (clean === '/login' || clean === '/register') return '/'
  if (!['/', '/markets', '/dashboard', '/admin', '/profile'].includes(clean)) return '/'
  return clean + rawSearch
}

function scrollToHash(hash: string) {
  // Deferred so landing sections exist when arriving from another page.
  requestAnimationFrame(() => {
    if (hash) document.getElementById(hash)?.scrollIntoView()
    else window.scrollTo(0, 0)
  })
}

function focusMain() {
  // Keyboard and screen-reader users start at the new page's content instead
  // of the header. preventScroll keeps the scroll position scrollToHash set.
  requestAnimationFrame(() => {
    const main = document.querySelector('main')
    if (main instanceof HTMLElement) main.focus({ preventScroll: true })
  })
}

class RouteErrorBoundary extends React.Component<{ onHome: () => void; children: React.ReactNode }, { error: string | null }> {
  state = { error: null as string | null }
  static getDerivedStateFromError(error: unknown) {
    return { error: error instanceof Error && error.message ? error.message : 'Something went wrong' }
  }
  render() {
    if (this.state.error) {
      return <main id="main" tabIndex={-1} className="notfound-main">
        <p className="eyebrow">ALPHASENSE</p>
        <h1>Something broke.</h1>
        <p>This section crashed ({this.state.error}). The rest of the site is unaffected.</p>
        <p><a href="/" onClick={event => { event.preventDefault(); this.setState({ error: null }); this.props.onHome() }}>Back to the research note</a></p>
      </main>
    }
    return this.props.children
  }
}

function GateNotice({ eyebrow, title, copy, action, id = 'main' }: { eyebrow: string; title: string; copy?: string; action?: React.ReactNode; id?: string }) {
  return <main id={id} tabIndex={-1} className="notfound-main" aria-busy="true" aria-live="polite">
    <p className="eyebrow">{eyebrow}</p>
    <h1>{title}</h1>
    {copy && <p>{copy}</p>}
    <div aria-hidden="true" className="mt-8 flex max-w-md flex-col gap-3">
      <span className="block h-3 w-3/4 animate-pulse rounded-sm bg-hairline" />
      <span className="block h-3 w-1/2 animate-pulse rounded-sm bg-hairline" />
      <span className="block h-3 w-2/3 animate-pulse rounded-sm bg-hairline" />
    </div>
    {action}
  </main>
}

function App() {
  const [route, setRoute] = useState(window.location.pathname + window.location.search + window.location.hash)
  const [user, setUser] = useState<AuthUser | null>(null)
  const [authLoading, setAuthLoading] = useState(true)
  const homeRef = useRef<HTMLElement>(null)
  const { pathname: rawPathname, search } = splitRoute(route)
  const pathname = normalizePathname(rawPathname)
  const authPage = pathname === '/login' || pathname === '/register'
  const marketPage = pathname === '/markets' || pathname === '/dashboard'
  const adminPage = pathname === '/admin'
  const profilePage = pathname === '/profile'
  const navigate = (next: string) => {
    const { pathname, search, hash } = splitRoute(next)
    const target = normalizePathname(pathname) + search + (hash ? `#${hash}` : '')
    const fromPathname = normalizePathname(splitRoute(window.location.pathname).pathname)
    const routeChange = normalizePathname(pathname) !== fromPathname
    if (window.location.pathname + window.location.search + window.location.hash !== target) window.history.pushState({}, '', target)
    setRoute(target)
    scrollToHash(hash)
    // Same-page hash jumps keep focus where it is; real route changes move it
    // to the new page's content.
    if (routeChange) focusMain()
  }
  useEffect(() => {
    const pop = () => setRoute(window.location.pathname + window.location.search + window.location.hash)
    window.addEventListener('popstate', pop)
    getCurrentUser().then(setUser).catch(() => setUser(null)).finally(() => setAuthLoading(false))
    // Honor a hash on first load (e.g. a shared /#cutoff link).
    const { hash } = splitRoute(window.location.pathname + window.location.search + window.location.hash)
    if (hash) scrollToHash(hash)
    return () => window.removeEventListener('popstate', pop)
  }, [])
  // Session heartbeat: if the cookie expires (or the user signs out in
  // another tab) while sitting on a private page, drop the stale identity
  // and send them back through sign-in instead of showing a dead dashboard.
  // Network blips are ignored — only an explicit anonymous response logs out.
  const userRef = useRef<AuthUser | null>(null)
  userRef.current = user
  const navigateRef = useRef(navigate)
  navigateRef.current = navigate
  useEffect(() => {
    const timer = window.setInterval(() => {
      getCurrentUser().then(current => {
        if (!current && userRef.current) {
          userRef.current = null
          setUser(null)
          const { pathname: here, search: hereSearch } = splitRoute(window.location.pathname + window.location.search)
          const clean = normalizePathname(here)
          if (clean === '/markets' || clean === '/dashboard' || clean === '/admin' || clean === '/profile') {
            navigateRef.current(`/login?next=${encodeURIComponent(clean + hereSearch)}`)
          }
        } else if (current && !userRef.current) {
          userRef.current = current
          setUser(current)
        }
      }).catch(() => { /* unreachable API: keep the current session */ })
    }, 5 * 60_000)
    return () => window.clearInterval(timer)
  }, [])
  useLayoutEffect(() => {
    if (pathname !== '/' || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const root = homeRef.current
    if (!root) return
    gsap.registerPlugin(ScrollTrigger)
    const context = gsap.context(() => {
      gsap.utils.toArray<HTMLElement>('.reveal-section', root).forEach(section => {
        const reveal = gsap.timeline({ scrollTrigger: { trigger: section, start: 'top 78%', once: true } })
        reveal.fromTo(section.querySelector('.section-heading h2'), { clipPath: 'inset(0 100% 0 0)' }, { clipPath: 'inset(0 0% 0 0)', duration: .7, ease: 'power2.inOut' })
          .fromTo(section.querySelector('.rail-rule'), { scaleY: 0, transformOrigin: 'center bottom' }, { scaleY: 1, duration: .65, ease: 'power2.out' }, .1)
          .fromTo(section.querySelector('.explorer'), { clipPath: 'inset(0 0 100% 0)' }, { clipPath: 'inset(0 0 0% 0)', duration: .75, ease: 'power2.inOut' }, .2)
          .fromTo(section.querySelector('.method-footnote'), { scaleX: .01, transformOrigin: 'left center' }, { scaleX: 1, duration: .65, ease: 'power2.out' }, .65)
      })
    }, root)
    return () => context.revert()
  }, [pathname])
  const handleSignOut = async () => { try { await signOut() } catch { /* clear local identity if the API is unreachable */ } finally { setUser(null); navigate('/') } }
  const next = safeNext(new URLSearchParams(search).get('next'))
  const goHome = (event: React.MouseEvent<HTMLAnchorElement>, path: string) => { event.preventDefault(); navigate(path) }
  // Before login: /markets, /admin and /profile are private. After login: /login redirects away.
  useEffect(() => {
    if (authLoading) return
    if (marketPage && !user) navigate(`/login?next=${encodeURIComponent(pathname + search)}`)
    if (adminPage && (!user || user.role !== 'admin')) navigate(`/login?next=${encodeURIComponent(pathname + search)}`)
    if (profilePage && !user) navigate(`/login?next=${encodeURIComponent(pathname + search)}`)
  }, [authLoading, marketPage, adminPage, profilePage, user, pathname, search])
  useEffect(() => {
    if (authLoading) return
    if (authPage && user) navigate(next !== '/' ? next : '/markets')
  }, [authLoading, authPage, user, next])
  if (authLoading) return <><a className="skip-link" href="#main">Skip to content</a><Header user={user} onNavigate={navigate} onSignOut={() => { void handleSignOut() }} /><GateNotice eyebrow="ALPHASENSE" title="Loading…" copy="Restoring your session." /></>
  if (authPage) {
    if (user) return <><a className="skip-link" href="#main">Skip to content</a><Header user={user} onNavigate={navigate} onSignOut={() => { void handleSignOut() }} /><GateNotice eyebrow="ALPHASENSE" title="Redirecting…" copy="You are already signed in." /></>
    return <RouteErrorBoundary key={route} onHome={() => navigate('/')}><a className="skip-link" href="#main">Skip to content</a><Header user={user} onNavigate={navigate} onSignOut={() => { void handleSignOut() }} /><Suspense fallback={<GateNotice eyebrow="ALPHASENSE / ACCOUNT" title="Loading…" copy="Preparing the sign-in form." />}><AuthPage mode={pathname === '/register' ? 'register' : 'login'} next={next} onNavigate={navigate} onSuccess={nextUser => { setUser(nextUser); navigate(next !== '/' ? next : '/markets') }} /></Suspense></RouteErrorBoundary>
  }
  if (marketPage) {
    if (!user) return <><a className="skip-link" href="#market-main">Skip to content</a><Header user={user} marketPage onNavigate={navigate} onSignOut={() => { void handleSignOut() }} /><GateNotice id="market-main" eyebrow="ALPHASENSE / ACCOUNT" title="Sign in required." copy="Market signals are private. Redirecting to sign in…" action={<p><a href="/login" onClick={event => { event.preventDefault(); navigate(`/login?next=${encodeURIComponent(pathname + search)}`) }}>Sign in →</a></p>} /></>
    return <RouteErrorBoundary key={route} onHome={() => navigate('/')}><a className="skip-link" href="#market-main">Skip to content</a><Header user={user} marketPage onNavigate={navigate} onSignOut={() => { void handleSignOut() }} /><Suspense fallback={<GateNotice id="market-main" eyebrow="ALPHASENSE / MARKETS" title="Loading…" copy="Fetching the latest market snapshot." />}><MarketDashboard onNavigate={navigate} search={search} /></Suspense></RouteErrorBoundary>
  }
  if (profilePage) {
    if (!user) return <><a className="skip-link" href="#main">Skip to content</a><Header user={user} onNavigate={navigate} onSignOut={() => { void handleSignOut() }} /><GateNotice eyebrow="ALPHASENSE / ACCOUNT" title="Sign in required." copy="Your profile is private. Redirecting to sign in…" action={<p><a href="/login" onClick={event => { event.preventDefault(); navigate(`/login?next=${encodeURIComponent(pathname + search)}`) }}>Sign in →</a></p>} /></>
    return <RouteErrorBoundary key={route} onHome={() => navigate('/')}><a className="skip-link" href="#main">Skip to content</a><Header user={user} onNavigate={navigate} onSignOut={() => { void handleSignOut() }} /><Suspense fallback={<GateNotice eyebrow="ALPHASENSE / ACCOUNT" title="Loading…" copy="Loading your profile." />}><ProfilePage user={user} onNavigate={navigate} onUserChange={setUser} /></Suspense></RouteErrorBoundary>
  }
  if (adminPage) {    if (!user || user.role !== 'admin') return <><a className="skip-link" href="#market-main">Skip to content</a><Header user={user} adminPage onNavigate={navigate} onSignOut={() => { void handleSignOut() }} /><GateNotice id="market-main" eyebrow="ALPHASENSE / ADMIN" title="Admin sign in required." copy="Administration is restricted. Redirecting to sign in…" action={<p><a href="/login" onClick={event => { event.preventDefault(); navigate(`/login?next=${encodeURIComponent(pathname + search)}`) }}>Sign in →</a></p>} /></>
    return <RouteErrorBoundary key={route} onHome={() => navigate('/')}><a className="skip-link" href="#market-main">Skip to content</a><Header user={user} adminPage onNavigate={navigate} onSignOut={() => { void handleSignOut() }} /><Suspense fallback={<GateNotice id="market-main" eyebrow="ALPHASENSE / ADMIN" title="Loading…" copy="Loading administration." />}><AdminDashboard /></Suspense></RouteErrorBoundary>
  }
  if (pathname !== '/') return <RouteErrorBoundary key={route} onHome={() => navigate('/')}><a className="skip-link" href="#main">Skip to content</a><Header user={user} onNavigate={navigate} onSignOut={() => { void handleSignOut() }} /><main id="main" tabIndex={-1} className="notfound-main"><p className="eyebrow">404 / NOT FOUND</p><h1>No such page.</h1><p>The address <code>{pathname}</code> is not part of this research site.</p><p><a href="/" onClick={event => goHome(event, '/')}>Research note</a><span aria-hidden="true"> · </span><a href="/markets" onClick={event => goHome(event, '/markets')}>Markets</a></p></main></RouteErrorBoundary>
  return <RouteErrorBoundary key={route} onHome={() => navigate('/')}><a className="skip-link" href="#main">Skip to content</a><Header user={user} onNavigate={navigate} onSignOut={() => { void handleSignOut() }} /><main ref={homeRef} id="main" tabIndex={-1}><Suspense fallback={<GateNotice eyebrow="ALPHASENSE / RESEARCH" title="Loading…" copy="Preparing the research note." />}><HomePage user={user} onNavigate={navigate} /></Suspense><LeakageExplorer /><Suspense fallback={null}><ClosingSection user={user} onNavigate={navigate} /></Suspense><SiteFooter user={user} onNavigate={navigate} /></main></RouteErrorBoundary>
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>)
