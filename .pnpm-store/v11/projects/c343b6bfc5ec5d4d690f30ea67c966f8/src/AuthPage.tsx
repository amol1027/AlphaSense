import { useEffect, useState, type FormEvent } from 'react'
import { ArrowLeft, ArrowRight, Eye, EyeOff } from 'lucide-react'
import { authenticate, getCsrfToken, type AuthUser } from './auth-client'

type AuthPageProps = {
  mode: 'login' | 'register'
  next: string
  onNavigate: (path: string) => void
  onSuccess: (user: AuthUser) => void
}

export function AuthPage({ mode, next, onNavigate, onSuccess }: AuthPageProps) {
  const registering = mode === 'register'
  const switchPath = `${registering ? '/login' : '/register'}${next === '/' ? '' : `?next=${encodeURIComponent(next)}`}`
  const [csrfToken, setCsrfToken] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [emailUpdates, setEmailUpdates] = useState(true)
  const [whatsappUpdates, setWhatsappUpdates] = useState(false)
  const [whatsappNumber, setWhatsappNumber] = useState('')

  useEffect(() => {
    let active = true
    getCsrfToken().then(token => { if (active) setCsrfToken(token) }).catch(() => {
      if (active) setError('The account service is unavailable. Start the AlphaSense account API and try again.')
    })
    return () => { active = false }
  }, [])

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError('')
    if (!csrfToken) {
      setError('The form is still preparing. Refresh this page and try again.')
      return
    }
    const form = new FormData(event.currentTarget)
    const values: Record<string, unknown> = {
      email: String(form.get('email') || '').trim(),
      password: String(form.get('password') || ''),
    }
    if (registering) {
      values.display_name = String(form.get('displayName') || '').trim()
      if (values.password !== String(form.get('confirmPassword') || '')) {
        setError('Those passwords do not match.')
        return
      }
      values.email_updates = emailUpdates
      values.whatsapp_updates = whatsappUpdates
      const cleanedWhatsapp = whatsappNumber.replace(/[\s\-()]/g, '').trim()
      if (cleanedWhatsapp) values.whatsapp_e164 = cleanedWhatsapp
      if (whatsappUpdates && !cleanedWhatsapp) {
        setError('Add your WhatsApp number to enable WhatsApp reminders.')
        return
      }
    }
    setBusy(true)
    try {
      const user = await authenticate(mode, csrfToken, values)
      onSuccess(user)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'We could not complete that request. Please try again.')
    } finally {
      setBusy(false)
    }
  }

  return <main className="auth-main" id="main" tabIndex={-1}>
    <a className="auth-back" href="/" onClick={event => { event.preventDefault(); onNavigate('/') }}><ArrowLeft size={14} /> Back to the research note</a>
    <div className="auth-layout">
      <div className="auth-intro">
        <p className="eyebrow">ALPHASENSE / ACCOUNT</p>
        <h1>{registering ? <>Create an<br /><em>account.</em></> : <>Welcome<br /><em>back.</em></>}</h1>
        <p className="auth-copy">{registering ? 'Set up an AlphaSense account for this research site.' : 'Sign in to your AlphaSense account.'}{next !== '/' && <> You’ll return to <strong>{next === '/markets' || next === '/dashboard' ? 'Markets' : 'the research note'}</strong> afterwards.</>}</p>
        <p className="auth-public-note"><span>01</span> The research note remains public. Your account does not change the research or its results.</p>
      </div>
      <section className="auth-form-panel" aria-labelledby="auth-form-heading">
        <div className="auth-panel-top"><span>{registering ? 'NEW ACCOUNT' : 'ACCOUNT ACCESS'}</span><span>IST · NSE / BSE</span></div>
        <h2 id="auth-form-heading">{registering ? 'Your details' : 'Sign in'}</h2>
        <form className="auth-form" onSubmit={submit}>
          {registering && <label className="form-field"><span>Name</span><input name="displayName" autoComplete="name" required maxLength={80} placeholder="Your name" /></label>}
          <label className="form-field"><span>Email</span><input name="email" type="email" autoComplete="email" required maxLength={254} placeholder="you@example.com" /></label>
          <label className="form-field"><span>Password</span><span className="password-input"><input name="password" type={showPassword ? 'text' : 'password'} autoComplete={registering ? 'new-password' : 'current-password'} required minLength={registering ? 10 : 1} maxLength={128} placeholder={registering ? 'At least 10 characters' : 'Your password'} /><button type="button" className="password-toggle" onClick={() => setShowPassword(value => !value)} aria-label={showPassword ? 'Hide password' : 'Show password'} aria-pressed={showPassword}>{showPassword ? <EyeOff size={16} /> : <Eye size={16} />}</button></span>{registering && <small>Use at least 10 characters.</small>}</label>
          {registering && <label className="form-field"><span>Confirm password</span><input name="confirmPassword" type={showPassword ? 'text' : 'password'} autoComplete="new-password" required minLength={10} maxLength={128} placeholder="Enter it once more" /></label>}
          {registering && <fieldset className="form-field" style={{ border: 0, padding: 0, margin: 0 }}>
            <span>Reminders (research notes, not trading advice)</span>
            <label className="hub-show"><input type="checkbox" checked={emailUpdates} onChange={event => setEmailUpdates(event.target.checked)} /> Email me high-range regime alerts</label>
            <label className="hub-show"><input type="checkbox" checked={whatsappUpdates} onChange={event => setWhatsappUpdates(event.target.checked)} /> WhatsApp alerts</label>
            {whatsappUpdates && <input name="whatsappNumber" value={whatsappNumber} onChange={event => setWhatsappNumber(event.target.value)} autoComplete="tel" inputMode="tel" maxLength={20} placeholder="+919876543210" aria-label="WhatsApp number in E.164 format" />}
            <small>you can change this later in Profile.</small>
          </fieldset>}
          {error && <p className="auth-error" role="alert">{error}</p>}
          <button className="auth-submit" type="submit" disabled={busy || !csrfToken}>{busy ? 'Please wait…' : registering ? 'Create account' : 'Sign in'}<ArrowRight size={15} /></button>
        </form>
        <div className="auth-switch"><span>{registering ? 'Already registered?' : 'New to AlphaSense?'}</span><button type="button" onClick={() => onNavigate(switchPath)}>{registering ? 'Sign in' : 'Create an account'}</button></div>
        <p className="auth-privacy">Passwords are stored as Argon2id hashes. They are never saved as readable text.</p>
      </section>
    </div>
    <div className="auth-bottomline"><span>RESEARCH PROTOTYPE · NOT TRADING ADVICE</span><span>ACCOUNT SERVICE / MYSQL</span></div>
  </main>
}
