<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'

import {
  clearQueryActivity,
  type QueryActivity,
  readQueryActivity,
} from '../services/activityStore'

const records = ref<QueryActivity[]>([])
const filter = ref('')
const selected = ref<QueryActivity | null>(null)

const refresh = () => {
  records.value = readQueryActivity()
  if (selected.value) {
    selected.value = records.value.find(
      (record) => record.id === selected.value?.id,
    ) || null
  }
}

const filteredRecords = computed(() => {
  const search = filter.value.trim().toLowerCase()
  if (!search) return records.value
  return records.value.filter((record) => (
    `${record.sessionId} ${record.query} ${record.status}`
      .toLowerCase()
      .includes(search)
  ))
})

const completed = computed(
  () => records.value.filter((record) => record.status === 'completed').length,
)
const failed = computed(
  () => records.value.filter((record) => record.status === 'failed').length,
)
const processing = computed(
  () => records.value.filter((record) => record.status === 'processing').length,
)
const successRate = computed(() => {
  const settled = completed.value + failed.value
  return settled ? Math.round((completed.value / settled) * 100) : 0
})
const p95Latency = computed(() => {
  const values = records.value
    .map((record) => record.latencyMs)
    .filter((value): value is number => typeof value === 'number')
    .sort((left, right) => left - right)
  if (!values.length) return 0
  return values[Math.ceil(values.length * 0.95) - 1]
})

const formatDate = (timestamp: number) => new Intl.DateTimeFormat('en-GB', {
  dateStyle: 'medium',
  timeStyle: 'short',
}).format(timestamp)

const formatLatency = (milliseconds?: number) => {
  if (milliseconds === undefined) return '—'
  if (milliseconds < 1000) return `${milliseconds} ms`
  return `${(milliseconds / 1000).toFixed(1)} s`
}

const clearRecords = () => {
  if (!window.confirm('Clear locally recorded query activity?')) return
  clearQueryActivity()
  selected.value = null
  refresh()
}

onMounted(() => {
  refresh()
  window.addEventListener('storage', refresh)
  window.addEventListener('lexguard-query-activity', refresh)
})

onUnmounted(() => {
  window.removeEventListener('storage', refresh)
  window.removeEventListener('lexguard-query-activity', refresh)
})
</script>

<template src="../../templates/legal-query-monitor.html"></template>
