const browserHost = window.location.hostname || '127.0.0.1'
const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:'

export const queryApiBase = (
  import.meta.env.VITE_QUERY_API_BASE_URL
  || `${protocol}//${browserHost}:8008`
).replace(/\/+$/, '')

export const importApiBase = (
  import.meta.env.VITE_IMPORT_API_BASE_URL
  || `${protocol}//${browserHost}:8007`
).replace(/\/+$/, '')

export const parseJsonResponse = async <T>(response: Response): Promise<T> => {
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    const message = body.detail || body.error || `Request failed (${response.status})`
    throw new Error(String(message))
  }
  return body as T
}

export const checkHealth = async (apiBase: string): Promise<boolean> => {
  try {
    const response = await fetch(`${apiBase}/health`, {
      signal: AbortSignal.timeout(4000),
    })
    return response.ok
  } catch {
    return false
  }
}
