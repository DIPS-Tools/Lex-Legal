<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { RouterLink, RouterView } from 'vue-router'

import { checkHealth, importApiBase, queryApiBase } from './services/api'

const mobileNavigationOpen = ref(false)
const queryOnline = ref(false)
const importOnline = ref(false)
let healthTimer: number | undefined

const refreshHealth = async () => {
  const [queryStatus, importStatus] = await Promise.all([
    checkHealth(queryApiBase),
    checkHealth(importApiBase),
  ])
  queryOnline.value = queryStatus
  importOnline.value = importStatus
}

onMounted(() => {
  refreshHealth()
  healthTimer = window.setInterval(refreshHealth, 15_000)
})

onUnmounted(() => window.clearInterval(healthTimer))
</script>

<template>
  <div class="site-shell">
    <header class="site-header">
      <RouterLink class="brand" to="/" @click="mobileNavigationOpen = false">
        <span class="brand-mark" aria-hidden="true">
          <svg viewBox="0 0 36 36">
            <path d="M10 8h16M18 8v20M8 14h20M10 14l-4 8h8l-4-8Zm16 0-4 8h8l-4-8Z" />
          </svg>
        </span>
        <span>
          <strong>LexGuard</strong>
          <small>Legal intelligence workspace</small>
        </span>
      </RouterLink>

      <button
        class="nav-toggle"
        type="button"
        aria-label="Toggle navigation"
        :aria-expanded="mobileNavigationOpen"
        @click="mobileNavigationOpen = !mobileNavigationOpen"
      >
        <span />
        <span />
      </button>

      <nav :class="{ open: mobileNavigationOpen }" aria-label="Primary navigation">
        <RouterLink to="/" @click="mobileNavigationOpen = false">Overview</RouterLink>
        <RouterLink to="/legal-compliance" @click="mobileNavigationOpen = false">Legal Query</RouterLink>
        <RouterLink to="/import" @click="mobileNavigationOpen = false">Knowledge Import</RouterLink>
        <RouterLink to="/legal-query-monitor" @click="mobileNavigationOpen = false">Monitor</RouterLink>
      </nav>

      <div class="service-cluster" aria-label="Backend service status">
        <span class="service-state" :class="{ online: queryOnline }">
          <i /> Query API
        </span>
        <span class="service-state" :class="{ online: importOnline }">
          <i /> Import API
        </span>
      </div>
    </header>

    <main class="page-main">
      <RouterView />
    </main>
  </div>
</template>
