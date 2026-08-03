<script setup lang="ts">
import { computed, ref } from 'vue'

import { importApiBase, parseJsonResponse } from '../services/api'

interface UploadTask {
  id: string
  file: File
  status: 'uploading' | 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  done: string[]
  running: string[]
  error?: string
}

interface UploadResponse {
  task_ids?: string[]
  tasks?: Array<{ task_id: string }>
}

interface StatusResponse {
  status: UploadTask['status']
  done_list?: string[]
  running_list?: string[]
  error?: string
}

const fileInput = ref<HTMLInputElement | null>(null)
const dragging = ref(false)
const tasks = ref<UploadTask[]>([])

const completedCount = computed(
  () => tasks.value.filter((task) => task.status === 'completed').length,
)

const delay = (milliseconds: number) => new Promise(
  (resolve) => window.setTimeout(resolve, milliseconds),
)

const formatBytes = (bytes: number) => {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

const pollTask = async (task: UploadTask) => {
  let failures = 0
  while (task.status !== 'completed' && task.status !== 'failed') {
    await delay(1600)
    try {
      const response = await fetch(
        `${importApiBase}/status/${encodeURIComponent(task.id)}`,
        { cache: 'no-store' },
      )
      const data = await parseJsonResponse<StatusResponse>(response)
      failures = 0
      task.status = data.status
      task.done = data.done_list || []
      task.running = data.running_list || []
      task.error = data.error || ''
      task.progress = data.status === 'completed'
        ? 100
        : Math.min(94, 18 + task.done.length * 10)
    } catch (error) {
      failures += 1
      if (failures >= 5) {
        task.status = 'failed'
        task.error = error instanceof Error ? error.message : String(error)
      }
    }
  }
}

const uploadFile = async (file: File) => {
  const extension = file.name.split('.').pop()?.toLowerCase()
  const task: UploadTask = {
    id: `pending-${Date.now()}-${file.name}`,
    file,
    status: 'uploading',
    progress: 8,
    done: [],
    running: ['Uploading file'],
  }
  tasks.value.unshift(task)

  if (!extension || !['pdf', 'md'].includes(extension)) {
    task.status = 'failed'
    task.error = 'Only PDF and Markdown files are supported.'
    return
  }

  const body = new FormData()
  body.append('files', file)
  try {
    const response = await fetch(`${importApiBase}/upload`, {
      method: 'POST',
      body,
    })
    const data = await parseJsonResponse<UploadResponse>(response)
    const taskId = data.tasks?.[0]?.task_id || data.task_ids?.[0]
    if (!taskId) throw new Error('The import service returned no task ID.')
    task.id = taskId
    task.status = 'pending'
    task.progress = 16
    task.done = ['Uploading file']
    task.running = []
    await pollTask(task)
  } catch (error) {
    task.status = 'failed'
    task.error = error instanceof Error ? error.message : String(error)
  }
}

const acceptFiles = (files: FileList | File[]) => {
  Array.from(files).forEach(uploadFile)
}

const handleDrop = (event: DragEvent) => {
  dragging.value = false
  if (event.dataTransfer?.files) acceptFiles(event.dataTransfer.files)
}

const handleFileInput = (event: Event) => {
  const input = event.target as HTMLInputElement
  if (input.files) acceptFiles(input.files)
  input.value = ''
}
</script>

<template src="../../templates/import.html"></template>
