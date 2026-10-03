import React from 'react'

interface SiteFooterProps {
  onNavigate: (path: string) => void
  user?: { displayName: string; email: string } | null
}

// Site-wide footer. Rendered last on the landing page — after HomePage and
// the temporal-cutoff section — so it always closes the document.
export function SiteFooter({ onNavigate, user }: SiteFooterProps) {
  const go = (e: React.MouseEvent<HTMLAnchorElement>, path: string) => { e.preventDefault(); onNavigate(path) }
  return (
    <footer className="border-t border-hairline">
      <div className="mx-auto grid w-full max-w-[1500px] gap-10 px-6 py-12 md:grid-cols-[1.4fr_1fr_1fr] md:px-24">
        <div>
          <a className="wordmark" href="/#top" onClick={e => go(e, '/#top')} aria-label="AlphaSense home"><span className="wordmark-mark" aria-hidden="true"><i>A</i><i>S</i></span><span>AlphaSense</span></a>
          <p className="mt-4 max-w-sm text-sm leading-relaxed text-muted">Short-horizon NSE research with a record of restraint. Range regime held across 130 held-out folds; direction did not.</p>
          <p className="mt-4 font-mono text-[8px] tracking-[0.09em] text-muted">15-MINUTE NSE DATA · IST · PHASE 4 PILOT</p>
        </div>
        <nav aria-label="Research">
          <p className="font-mono text-[9px] tracking-[0.14em] text-muted">RESEARCH</p>
          <ul className="mt-4 flex flex-col gap-2.5 text-sm">
            <li><a className="text-ink hover:text-accent" href="#findings" onClick={e => go(e, '/#findings')}>Findings</a></li>
            <li><a className="text-ink hover:text-accent" href="#methodology" onClick={e => go(e, '/#methodology')}>Methodology</a></li>
            <li><a className="text-ink hover:text-accent" href="#cutoff" onClick={e => go(e, '/#cutoff')}>Temporal cutoff</a></li>
            <li><a className="text-ink hover:text-accent" href="#faq" onClick={e => go(e, '/#faq')}>FAQ</a></li>
          </ul>
        </nav>
        <nav aria-label="Workspace">
          <p className="font-mono text-[9px] tracking-[0.14em] text-muted">WORKSPACE</p>
          <ul className="mt-4 flex flex-col gap-2.5 text-sm">
            {user
              ? <><li><a className="text-ink hover:text-accent" href="/markets" onClick={e => go(e, '/markets')}>Open dashboard</a></li><li><a className="text-ink hover:text-accent" href="#about" onClick={e => go(e, '/#about')}>About</a></li></>
              : <><li><a className="text-ink hover:text-accent" href="/login" onClick={e => go(e, '/login?next=%2Fmarkets')}>Sign in</a></li><li><a className="text-ink hover:text-accent" href="/register" onClick={e => go(e, '/register?next=%2Fmarkets')}>Create account</a></li><li><a className="text-ink hover:text-accent" href="/markets" onClick={e => go(e, '/markets')}>Market workspace</a></li></>}
          </ul>
        </nav>
      </div>
      <div className="border-t border-hairline">
        <div className="mx-auto flex w-full max-w-[1500px] flex-wrap items-center justify-between gap-2 px-6 py-4 font-mono text-[8px] tracking-[0.07em] text-muted md:px-24">
          <span>© 2026 ALPHASENSE · EXPLORATORY · NOT INVESTMENT ADVICE</span>
          <a className="hover:text-accent" href="#top" onClick={e => go(e, '/#top')}>BACK TO TOP ↑</a>
        </div>
      </div>
    </footer>
  )
}
