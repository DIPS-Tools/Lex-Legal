export type QueryStatus = 'processing' | 'completed' | 'failed'

export interface QueryActivity {
  id: string
  sessionId: string
  query: string
  status: QueryStatus
  startedAt: number
  updatedAt: number
  latencyMs?: number
  answerLength: number
  doneStages: string[]
  runningStages: string[]
  error?: string
}

const STORAGE_KEY = 'lexguard_query_activity'
const MAX_RECORDS = 200

export const readQueryActivity = (): QueryActivity[] => {
  try {
    const value = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
    return Array.isArray(value) ? value : []
  } catch {
    return []
  }
}

export const upsertQueryActivity = (
  activity: QueryActivity,
): QueryActivity[] => {
  const records = readQueryActivity()
  const index = records.findIndex(
    (record) => record.id === activity.id,
  )
  if (index >= 0) records[index] = activity
  else records.unshift(activity)
  const limited = records
    .sort((left, right) => right.updatedAt - left.updatedAt)
    .slice(0, MAX_RECORDS)
  localStorage.setItem(STORAGE_KEY, JSON.stringify(limited))
  window.dispatchEvent(new CustomEvent('lexguard-query-activity'))
  return limited
}

export const clearQueryActivity = (): void => {
  localStorage.removeItem(STORAGE_KEY)
  window.dispatchEvent(new CustomEvent('lexguard-query-activity'))
}
