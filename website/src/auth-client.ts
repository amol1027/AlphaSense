export type AuthUser = { id: number; email: string; displayName: string; role?: string }

const apiBase = (import.meta.env.VITE_AUTH_API_BASE_URL || '/auth-api').replace(/\/$/, '')
let csrfRequest: Promise<string> | null = null

async function readError(response: Response): Promise<Error> {
  try {
    const data = await response.json()
    const detail = data?.detail
    if (typeof detail === 'string') return new Error(detail)
    if (Array.isArray(detail) && detail[0]?.msg) return new Error(String(detail[0].msg).replace(/^Value error, /, ''))
  } catch { /* use the status fallback */ }
  return new Error(`Account request failed (${response.status}).`)
}

export async function getCsrfToken(): Promise<string> {
  if (!csrfRequest) {
    csrfRequest = (async () => {
      const response = await fetch(`${apiBase}/api/auth/csrf`, { credentials: 'include' })
      if (!response.ok) throw await readError(response)
      return (await response.json()).csrfToken as string
    })().catch(error => {
      csrfRequest = null
      throw error
    })
  }
  return csrfRequest
}

export async function getCurrentUser(): Promise<AuthUser | null> {
  const response = await fetch(`${apiBase}/api/auth/me`, { credentials: 'include' })
  if (response.status === 401) return null
  if (!response.ok) throw await readError(response)
  return await response.json() as AuthUser
}

export async function authenticate(mode: 'login' | 'register', csrfToken: string, values: Record<string, unknown>): Promise<AuthUser> {
  // Refresh immediately before a state-changing request so an older token
  // held by an already-open page cannot disagree with the current cookie.
  csrfRequest = null
  let currentCsrfToken = await getCsrfToken()
  const send = (token: string) => fetch(`${apiBase}/api/auth/${mode}`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': token || csrfToken },
    body: JSON.stringify(values),
  })
  let response = await send(currentCsrfToken)
  if (response.status === 403) {
    const error = await readError(response)
    if (error.message !== 'Refresh the page and try again') throw error
    // A stale session-bound token can outlive a page or another tab. Refresh
    // both sides of the double-submit pair, then retry the rejected request once.
    csrfRequest = null
    currentCsrfToken = await getCsrfToken()
    response = await send(currentCsrfToken)
  }
  if (!response.ok) throw await readError(response)
  // Login and registration rotate both cookies; fetch the session-bound token
  // again for a later logout request.
  csrfRequest = null
  return await response.json() as AuthUser
}

export async function signOut(): Promise<void> {
  csrfRequest = null
  const csrfToken = await getCsrfToken()
  const response = await fetch(`${apiBase}/api/auth/logout`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'X-CSRF-Token': csrfToken },
  })
  if (!response.ok) throw await readError(response)
}

export async function updateProfile(patch: { display_name?: string; email?: string }): Promise<AuthUser> {
  csrfRequest = null
  const csrfToken = await getCsrfToken()
  const response = await fetch(`${apiBase}/api/auth/me`, {
    method: 'PATCH',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': csrfToken },
    body: JSON.stringify(patch),
  })
  if (!response.ok) throw await readError(response)
  return await response.json() as AuthUser
}

export async function updateDisplayName(displayName: string): Promise<AuthUser> {
  return updateProfile({ display_name: displayName })
}

export async function updateEmail(email: string): Promise<AuthUser> {
  return updateProfile({ email })
}

export async function changePassword(currentPassword: string, newPassword: string): Promise<void> {
  csrfRequest = null
  const csrfToken = await getCsrfToken()
  const response = await fetch(`${apiBase}/api/auth/me/password`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': csrfToken },
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
  })
  if (!response.ok) throw await readError(response)
}

export type NotificationPrefs = {
  emailEnabled: boolean
  whatsappEnabled: boolean
  whatsappNumber: string | null
  assets: string[]
  minProbability: number
  dailySummary: boolean
}

export async function getNotificationPrefs(): Promise<NotificationPrefs> {
  const response = await fetch(`${apiBase}/api/auth/me/notifications`, { credentials: 'include' })
  if (!response.ok) throw await readError(response)
  return (await response.json()) as NotificationPrefs
}

export async function updateNotificationPrefs(patch: {
  email_enabled?: boolean
  whatsapp_enabled?: boolean
  whatsapp_e164?: string | null
  assets?: string[]
  min_probability?: number
  daily_summary?: boolean
}): Promise<NotificationPrefs> {
  csrfRequest = null
  const csrfToken = await getCsrfToken()
  const response = await fetch(`${apiBase}/api/auth/me/notifications`, {
    method: 'PATCH',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': csrfToken },
    body: JSON.stringify(patch),
  })
  if (!response.ok) throw await readError(response)
  return (await response.json()) as NotificationPrefs
}

export type AdminUser = {
  id: number; email: string; displayName: string; role: string;
  disabled: boolean; disabledAt: string | null; createdAt: string | null
}
export type AdminSession = { tokenHash: string; userId: number; expiresAt: string | null; createdAt: string | null }
export type AdminShadowRow = {
  asset: string | null; predictionTimestamp: string | null; probability: number | null;
  prediction: number | null; realizedLabel: number | null; correct: boolean | null
}
export type SystemStatus = {
  authDb: { ok: boolean }
  prediction: {
    ok: boolean; httpStatus: number | null; status: string | null;
    feedState: string | null; feedAgeMinutes: number | null;
    checks: Record<string, boolean>; error: string | null
  }
  marketCache: { present: boolean; rows: number | null; mtime: string | null; refresh: Record<string, unknown> | null }
  newsCache: { present: boolean; rows: number | null; mtime: string | null; refresh: Record<string, unknown> | null }
  shadowLog: { present: boolean; rows: number | null; mtime: string | null }
  modelArtifacts: { ok: boolean; present: string[]; missing: string[] }
  upstoxTokenConfigured: { ok: boolean }
  webapp: { ok: boolean }
}
export type SystemConfig = {
  assets: string[]; horizon: string; session: string; timezone: string;
  marketCacheRetentionDays: number; model: string; predictionHealthUrl: string;
  allowedOrigins: string[]; cookieSameSite: string; cookieSecure: boolean
}
export type SystemRefreshResult = { assetsUpdated: number; outcomesScored: number; predictionsAdded: number }
export type DayCount = { date: string; count: number }
export type AuditDay = { date: string; user: number; session: number; system: number; other: number }
export type ShadowDay = { date: string; logged: number; scored: number; correct: number }
export type RollingPoint = { date: string; accuracy: number | null }
export type AssetStat = { asset: string; logged: number; scored: number; correct: number; accuracy: number | null }
export type AnalyticsOverview = {
  users: { total: number; active: number; disabled: number; admins: number; signupsPerDay: DayCount[] }
  sessions: { live: number; perDay: DayCount[]; avgPerUser: number }
  audit: { last24h: number; perDay: AuditDay[]; topActors: { actorId: number; email: string | null; count: number }[] }
  shadow: {
    total: number; scored: number; correct: number; pending: number; accuracy: number | null
    perDay: ShadowDay[]; rolling7d: RollingPoint[]; perAsset: AssetStat[]; confidence: number[]
  }
}
export type AdminAuditEntry = {
  id: number; actorId: number; action: string; targetUserId: number | null;
  detail: string | null; createdAt: string | null
}

async function adminRead(path: string): Promise<unknown> {
  const response = await fetch(`${apiBase}${path}`, { credentials: 'include' })
  if (!response.ok) throw await readError(response)
  return await response.json()
}

async function adminWrite(path: string, method: 'PATCH' | 'POST' | 'DELETE', body?: unknown): Promise<unknown> {
  csrfRequest = null
  const csrfToken = await getCsrfToken()
  const response = await fetch(`${apiBase}${path}`, {
    method,
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': csrfToken },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!response.ok) throw await readError(response)
  return await response.json()
}

export const adminApi = {
  listUsers: (params: { search?: string; limit?: number; offset?: number; include_disabled?: boolean } = {}) => {
    const query = new URLSearchParams()
    if (params.search) query.set('search', params.search)
    query.set('limit', String(params.limit ?? 25))
    query.set('offset', String(params.offset ?? 0))
    if (params.include_disabled) query.set('include_disabled', 'true')
    return adminRead(`/api/admin/users?${query}`) as Promise<{ users: AdminUser[]; total: number; limit: number; offset: number }>
  },
  updateUser: (id: number, patch: { display_name?: string; role?: string; disabled?: boolean }) =>
    adminWrite(`/api/admin/users/${id}`, 'PATCH', patch) as Promise<AdminUser>,
  revokeUserSessions: (id: number) =>
    adminWrite(`/api/admin/users/${id}/revoke-sessions`, 'POST') as Promise<{ revoked: boolean }>,
  listSessions: (userId?: number) =>
    adminRead(userId ? `/api/admin/sessions?user_id=${userId}&limit=25` : '/api/admin/sessions?limit=25') as Promise<{ sessions: AdminSession[] }>,
  revokeSession: (tokenHash: string) =>
    adminWrite(`/api/admin/sessions/${tokenHash}`, 'DELETE') as Promise<{ revoked: boolean }>,
  listAudit: (offset = 0, limit = 25) =>
    adminRead(`/api/admin/audit?limit=${limit}&offset=${offset}`) as Promise<{ entries: AdminAuditEntry[] }>,
  systemStatus: () =>
    adminRead('/api/admin/system/status') as Promise<SystemStatus>,
  systemShadow: (limit = 50) =>
    adminRead(`/api/admin/system/shadow?limit=${limit}`) as Promise<{
      available: boolean; rows: AdminShadowRow[];
      counts: { scored: number; correct: number; pending: number }; limit: number
    }>,
  systemConfig: () =>
    adminRead('/api/admin/system/config') as Promise<SystemConfig>,
  systemRefresh: () =>
    adminWrite('/api/admin/system/refresh', 'POST') as Promise<SystemRefreshResult>,
  analytics: () =>
    adminRead('/api/admin/analytics/overview') as Promise<AnalyticsOverview>,
}

export async function downloadAdminCsv(path: '/api/admin/export/users' | '/api/admin/export/audit' | '/api/admin/export/shadow'): Promise<string> {
  const response = await fetch(`${apiBase}${path}`, { credentials: 'include' })
  if (!response.ok) throw await readError(response)
  const blob = await response.blob()
  const disposition = response.headers.get('content-disposition') ?? ''
  const filename = disposition.match(/filename="([^"]+)"/)?.[1] ?? 'export.csv'
  const url = URL.createObjectURL(blob)
  try {
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    link.remove()
  } finally {
    window.setTimeout(() => URL.revokeObjectURL(url), 4000)
  }
  return filename
}
