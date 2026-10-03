import { useSyncExternalStore } from 'react'

const STORAGE_KEY = 'alphasense-theme'

type Listener = () => void

const listeners = new Set<Listener>()

let dark = readInitial()
applyDom(dark)

function readStored(): boolean | null {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (raw === 'dark') return true
    if (raw === 'light') return false
  } catch {
    /* private mode etc. — fall through to OS preference */
  }
  return null
}

function readInitial(): boolean {
  const stored = readStored()
  if (stored !== null) return stored
  return window.matchMedia('(prefers-color-scheme: dark)').matches
}

function applyDom(value: boolean) {
  document.documentElement.dataset.theme = value ? 'dark' : 'light'
}

function emit() {
  applyDom(dark)
  for (const listener of listeners) listener()
}

function subscribe(listener: Listener) {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

function getSnapshot() {
  return dark
}

function getServerSnapshot() {
  return false
}

function setDark(next: boolean | ((value: boolean) => boolean)) {
  const value = typeof next === 'function' ? (next as (value: boolean) => boolean)(dark) : next
  if (value === dark) return
  dark = value
  try {
    window.localStorage.setItem(STORAGE_KEY, value ? 'dark' : 'light')
  } catch {
    /* keep the in-memory value when storage is unavailable */
  }
  emit()
}

// Follow the OS while the user has never chosen explicitly; once stored,
// the stored choice wins until the toggle changes it.
if (typeof window !== 'undefined' && readStored() === null) {
  const query = window.matchMedia('(prefers-color-scheme: dark)')
  const onChange = (event: MediaQueryListEvent) => {
    if (readStored() !== null) return
    if (event.matches !== dark) {
      dark = event.matches
      emit()
    }
  }
  if (typeof query.addEventListener === 'function') query.addEventListener('change', onChange)
  else query.addListener(onChange)
}

// Another tab changing the toggle syncs this tab.
window.addEventListener('storage', event => {
  if (event.key !== STORAGE_KEY) return
  const stored = readStored()
  const value = stored ?? window.matchMedia('(prefers-color-scheme: dark)').matches
  if (value !== dark) {
    dark = value
    emit()
  }
})

/** Shared dark-mode state: survives route changes, reloads, and syncs tabs. */
export function useTheme(): [boolean, (next: boolean | ((value: boolean) => boolean)) => void] {
  const value = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot)
  return [value, setDark]
}
