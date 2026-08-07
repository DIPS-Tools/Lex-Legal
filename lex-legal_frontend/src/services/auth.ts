import { reactive } from 'vue'

import { apiBase, parseJsonResponse } from './api'


export interface SessionUser {
  subject: string
  username: string
  email: string
  roles: string[]
  groups: string[]
  is_admin: boolean
}

interface LoginResponse {
  access_token: string
  expires_at: number
  user: SessionUser
}

type AuthStatus = 'checking' | 'anonymous' | 'authenticated'

const TOKEN_KEY = 'lexguard_access_token'
const EXPIRY_KEY = 'lexguard_access_expires_at'

export const authState = reactive<{
  status: AuthStatus
  token: string
  expiresAt: number
  user: SessionUser | null
}>({
  status: 'checking',
  token: localStorage.getItem(TOKEN_KEY) || '',
  expiresAt: Number(localStorage.getItem(EXPIRY_KEY) || 0),
  user: null,
})

let restorePromise: Promise<SessionUser | null> | null = null

export const clearAuthentication = () => {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(EXPIRY_KEY)
  localStorage.removeItem('lexguard_session_id')
  localStorage.removeItem('lexguard_query_activity')
  authState.token = ''
  authState.expiresAt = 0
  authState.user = null
  authState.status = 'anonymous'
}

const saveAuthentication = (response: LoginResponse) => {
  authState.token = response.access_token
  authState.expiresAt = Number(response.expires_at || 0)
  authState.user = response.user
  authState.status = 'authenticated'
  localStorage.setItem(TOKEN_KEY, authState.token)
  localStorage.setItem(EXPIRY_KEY, String(authState.expiresAt))
}

export const authorizationHeaders = (): Record<string, string> => (
  authState.token
    ? { Authorization: `Bearer ${authState.token}` }
    : {}
)

export const login = async (
  username: string,
  password: string,
): Promise<SessionUser> => {
  const response = await fetch(`${apiBase}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: username.trim(), password }),
  })
  const data = await parseJsonResponse<LoginResponse>(response)
  saveAuthentication(data)
  return data.user
}

export const restoreAuthentication = async (): Promise<SessionUser | null> => {
  if (authState.status === 'authenticated' && authState.user) {
    return authState.user
  }
  if (restorePromise) return restorePromise
  restorePromise = (async () => {
    if (!authState.token) {
      authState.status = 'anonymous'
      return null
    }
    if (authState.expiresAt && Date.now() >= authState.expiresAt) {
      clearAuthentication()
      return null
    }
    try {
      const response = await fetch(`${apiBase}/api/auth/me`, {
        headers: authorizationHeaders(),
      })
      const user = await parseJsonResponse<SessionUser>(response)
      authState.user = user
      authState.status = 'authenticated'
      return user
    } catch {
      clearAuthentication()
      return null
    }
  })().finally(() => {
    restorePromise = null
  })
  return restorePromise
}

export const logout = async (): Promise<void> => {
  if (authState.token) {
    try {
      await fetch(`${apiBase}/api/auth/logout`, {
        method: 'POST',
        headers: authorizationHeaders(),
      })
    } catch {
      // Local cleanup still logs out when either API or Keycloak is offline.
    }
  }
  clearAuthentication()
}

export const authenticatedFetch = async (
  input: RequestInfo | URL,
  init: RequestInit = {},
): Promise<Response> => {
  const headers = new Headers(init.headers)
  for (const [name, value] of Object.entries(authorizationHeaders())) {
    headers.set(name, value)
  }
  const response = await fetch(input, { ...init, headers })
  if (response.status === 401) {
    clearAuthentication()
    window.dispatchEvent(new CustomEvent('lexguard-authentication-required'))
  }
  return response
}
