<script setup lang="ts">
import DOMPurify from 'dompurify'
import { marked } from 'marked'
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  reactive,
  ref,
} from 'vue'
import { RouterLink } from 'vue-router'

import {
  consumeEventStream,
  parseJsonResponse,
  apiBase,
} from '../services/api'
import { authenticatedFetch } from '../services/auth'
import {
  type QueryActivity,
  upsertQueryActivity,
} from '../services/activityStore'

interface ProgressState {
  status: string
  done: string[]
  running: string[]
}

interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  pending?: boolean
  progress?: ProgressState
  error?: string
  imageUrls?: string[]
}

interface QueryResponse {
  answer?: string
  session_id: string
  done_list?: string[]
  error?: string
  image_urls?: string[]
}

interface HistoryResponse {
  items?: Array<{
    _id?: string
    role: string
    text: string
    ts?: string | number
    image_urls?: string[]
  }>
}

const makeId = () => globalThis.crypto?.randomUUID?.()
  ?? `${Date.now()}-${Math.random().toString(16).slice(2)}`

const sessionId = ref(
  localStorage.getItem('lexguard_session_id') || `legal-${makeId()}`,
)
localStorage.setItem('lexguard_session_id', sessionId.value)

const queryText = ref('')
const streamingEnabled = ref(true)
const sending = ref(false)
const historyLoading = ref(true)
const messages = ref<ChatMessage[]>([])
const chatEnd = ref<HTMLElement | null>(null)
const textarea = ref<HTMLTextAreaElement | null>(null)
let activeStreamController: AbortController | null = null

marked.setOptions({ breaks: true })

const renderedMessages = computed(() => messages.value.map((message) => ({
  ...message,
  html: message.role === 'assistant'
    ? DOMPurify.sanitize(marked.parse(message.content || '') as string)
    : '',
})))

const formatTime = (timestamp: number) => new Intl.DateTimeFormat('en-GB', {
  hour: '2-digit',
  minute: '2-digit',
}).format(timestamp)

const scrollToBottom = async () => {
  await nextTick()
  chatEnd.value?.scrollIntoView({ behavior: 'smooth', block: 'end' })
}

const resizeTextarea = async () => {
  await nextTick()
  if (!textarea.value) return
  textarea.value.style.height = '24px'
  textarea.value.style.height = `${Math.min(textarea.value.scrollHeight, 150)}px`
}

const updateActivity = (
  activityId: string,
  query: string,
  status: QueryActivity['status'],
  progress: ProgressState,
  startedAt: number,
  answerLength = 0,
  error = '',
) => {
  const now = Date.now()
  upsertQueryActivity({
    id: activityId,
    sessionId: sessionId.value,
    query,
    status,
    startedAt,
    updatedAt: now,
    latencyMs: status === 'processing' ? undefined : now - startedAt,
    answerLength,
    doneStages: progress.done,
    runningStages: progress.running,
    error: error || undefined,
  })
}

const loadHistory = async () => {
  historyLoading.value = true
  try {
    const response = await authenticatedFetch(
      `${apiBase}/api/query/history/${encodeURIComponent(sessionId.value)}?limit=30`,
    )
    const data = await parseJsonResponse<HistoryResponse>(response)
    messages.value = (data.items || []).map((item) => ({
      id: item._id || makeId(),
      role: item.role === 'user' ? 'user' : 'assistant',
      content: item.text || '',
      timestamp: item.ts ? new Date(item.ts).getTime() : Date.now(),
      imageUrls: item.image_urls || [],
    }))
  } catch {
    messages.value = []
  } finally {
    historyLoading.value = false
    scrollToBottom()
  }
}

const startNewConversation = async () => {
  activeStreamController?.abort()
  activeStreamController = null
  messages.value = []
  queryText.value = ''
  sending.value = false
  sessionId.value = `legal-${makeId()}`
  localStorage.setItem('lexguard_session_id', sessionId.value)
  await nextTick()
  textarea.value?.focus()
}

const finishAssistantMessage = (
  message: ChatMessage,
  activityId: string,
  submittedQuery: string,
  answer: string,
  progress: ProgressState,
  startedAt: number,
  error = '',
  imageUrls: string[] = [],
) => {
  message.pending = false
  message.content = answer
  message.error = error
  message.imageUrls = imageUrls
  message.progress = progress
  sending.value = false
  updateActivity(
    activityId,
    submittedQuery,
    error ? 'failed' : 'completed',
    progress,
    startedAt,
    answer.length,
    error,
  )
  scrollToBottom()
}

const openEventStream = async (
  assistantMessage: ChatMessage,
  activityId: string,
  submittedQuery: string,
  startedAt: number,
) => {
  let streamedAnswer = ''
  const progress: ProgressState = {
    status: 'processing',
    done: [],
    running: [],
  }

  activeStreamController?.abort()
  const controller = new AbortController()
  activeStreamController = controller
  try {
    const response = await authenticatedFetch(
      `${apiBase}/api/query/stream/${encodeURIComponent(sessionId.value)}`,
      {
        headers: { Accept: 'text/event-stream' },
        signal: controller.signal,
      },
    )
    if (!response.ok) {
      const error = await parseJsonResponse<{ detail?: string }>(response)
      throw new Error(error.detail || 'Unable to open the event stream.')
    }
    await consumeEventStream(response, ({ event, data }) => {
      const payload = JSON.parse(data || '{}')
      if (event === 'progress') {
        progress.status = payload.status || progress.status
        progress.done = Array.isArray(payload.done_list) ? payload.done_list : []
        progress.running = Array.isArray(payload.running_list)
          ? payload.running_list
          : []
        assistantMessage.progress = { ...progress }
        updateActivity(
          activityId,
          submittedQuery,
          payload.status === 'completed' ? 'completed' : 'processing',
          progress,
          startedAt,
          assistantMessage.content.length,
        )
        if (payload.status === 'completed' && assistantMessage.pending) {
          finishAssistantMessage(
            assistantMessage,
            activityId,
            submittedQuery,
            streamedAnswer || 'The workflow completed without returning answer text.',
            progress,
            startedAt,
          )
        }
      } else if (event === 'delta') {
        streamedAnswer += payload.delta || ''
        assistantMessage.content = streamedAnswer
        scrollToBottom()
      } else if (event === 'final' || event === 'final_answer') {
        finishAssistantMessage(
          assistantMessage,
          activityId,
          submittedQuery,
          payload.answer || streamedAnswer,
          progress,
          startedAt,
          '',
          payload.image_urls || [],
        )
      } else if (event === 'error') {
        throw new Error(payload.error || 'The query workflow failed.')
      }
    })
  } catch (error) {
    if (controller.signal.aborted) return
    finishAssistantMessage(
      assistantMessage,
      activityId,
      submittedQuery,
      streamedAnswer,
      progress,
      startedAt,
      error instanceof Error
        ? error.message
        : 'The streaming connection was interrupted.',
    )
  } finally {
    if (activeStreamController === controller) activeStreamController = null
  }
}

const sendQuery = async (suggestion?: string) => {
  const submittedQuery = (suggestion || queryText.value).trim()
  if (!submittedQuery || sending.value) return

  queryText.value = ''
  resizeTextarea()
  const startedAt = Date.now()
  const activityId = makeId()
  const initialProgress: ProgressState = {
    status: 'processing',
    done: [],
    running: ['Preparing legal research'],
  }
  // Keep the object reactive because SSE handlers mutate this same reference
  // after it has been inserted into the messages array. A plain object would
  // update in memory without invalidating renderedMessages, leaving the GUI
  // stuck on its initial "Preparing legal research" state.
  const assistantMessage = reactive<ChatMessage>({
    id: makeId(),
    role: 'assistant',
    content: '',
    timestamp: startedAt,
    pending: true,
    progress: initialProgress,
  })
  messages.value.push(
    {
      id: makeId(),
      role: 'user',
      content: submittedQuery,
      timestamp: startedAt,
    },
    assistantMessage,
  )
  sending.value = true
  updateActivity(
    activityId,
    submittedQuery,
    'processing',
    initialProgress,
    startedAt,
  )
  scrollToBottom()

  try {
    const response = await authenticatedFetch(`${apiBase}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: submittedQuery,
        session_id: sessionId.value,
        is_stream: streamingEnabled.value,
      }),
    })
    const data = await parseJsonResponse<QueryResponse>(response)
    sessionId.value = data.session_id || sessionId.value
    localStorage.setItem('lexguard_session_id', sessionId.value)

    if (streamingEnabled.value) {
      openEventStream(
        assistantMessage,
        activityId,
        submittedQuery,
        startedAt,
      )
      return
    }

    finishAssistantMessage(
      assistantMessage,
      activityId,
      submittedQuery,
      data.answer || '',
      {
        status: 'completed',
        done: data.done_list || [],
        running: [],
      },
      startedAt,
      data.error || '',
      data.image_urls || [],
    )
  } catch (error) {
    finishAssistantMessage(
      assistantMessage,
      activityId,
      submittedQuery,
      '',
      initialProgress,
      startedAt,
      error instanceof Error ? error.message : String(error),
    )
  }
}

const handleComposerKeydown = (event: KeyboardEvent) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    sendQuery()
  }
}

onMounted(loadHistory)
onBeforeUnmount(() => activeStreamController?.abort())
</script>

<template src="../../templates/legal-compliance.html"></template>
