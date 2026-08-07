const browserHost = window.location.hostname || '127.0.0.1'
const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:'

export const apiBase = (
  import.meta.env.VITE_API_BASE_URL
  || `${protocol}//${browserHost}:8008`
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

export interface ServerSentEvent {
  event: string
  data: string
}

export const consumeEventStream = async (
  response: Response,
  onEvent: (event: ServerSentEvent) => void,
): Promise<void> => {
  if (!response.body) throw new Error('The server returned no event stream.')
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  const consumeBlock = (block: string) => {
    let event = 'message'
    const data: string[] = []
    for (const line of block.split(/\r?\n/)) {
      if (line.startsWith('event:')) event = line.slice(6).trim()
      else if (line.startsWith('data:')) data.push(line.slice(5).trimStart())
    }
    if (data.length) onEvent({ event, data: data.join('\n') })
  }

  while (true) {
    const { value, done } = await reader.read()
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done })
    const blocks = buffer.split(/\r?\n\r?\n/)
    buffer = blocks.pop() || ''
    blocks.forEach(consumeBlock)
    if (done) break
  }
  if (buffer.trim()) consumeBlock(buffer)
}
