<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { RouterLink, RouterView } from 'vue-router'
import { useRouter } from 'vue-router'

import { apiBase, checkHealth } from './services/api'
import {
  authState,
  logout,
  restoreAuthentication,
} from './services/auth'

const router = useRouter()
const mobileNavigationOpen = ref(false)
const apiOnline = ref(false)
let healthTimer: number | undefined

const refreshHealth = async () => {
  apiOnline.value = await checkHealth(apiBase)
}

const signOut = async () => {
  mobileNavigationOpen.value = false
  await logout()
  await router.push('/')
}

const requireLogin = () => {
  router.push({ name: 'login' })
}

const handleAuthenticationRequired = () => {
  router.push({
    name: 'login',
    query: { redirect: router.currentRoute.value.fullPath },
  })
}

onMounted(() => {
  restoreAuthentication()
  refreshHealth()
  healthTimer = window.setInterval(refreshHealth, 15_000)
  window.addEventListener(
    'lexguard-authentication-required',
    handleAuthenticationRequired,
  )
})

onUnmounted(() => {
  window.clearInterval(healthTimer)
  window.removeEventListener(
    'lexguard-authentication-required',
    handleAuthenticationRequired,
  )
})
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
        <RouterLink
          v-if="authState.status === 'authenticated'"
          to="/legal-compliance"
          @click="mobileNavigationOpen = false"
        >Legal Compliance</RouterLink>
        <RouterLink
          v-if="authState.user?.is_admin"
          to="/import"
          @click="mobileNavigationOpen = false"
        >Knowledge Import</RouterLink>
        <RouterLink
          v-if="authState.user?.is_admin"
          to="/legal-query-monitor"
          @click="mobileNavigationOpen = false"
        >Monitor</RouterLink>
      </nav>

      <div class="header-actions">
        <div class="service-cluster" aria-label="Backend service status">
        <span class="service-state" :class="{ online: apiOnline }">
          <i /> Lex-Legal API
        </span>
        </div>
        <button
          v-if="authState.status !== 'authenticated'"
          class="header-login"
          type="button"
          @click="requireLogin"
        >Sign in</button>
        <button
          v-else
          class="header-user"
          type="button"
          :title="`Signed in as ${authState.user?.username}. Click to sign out.`"
          @click="signOut"
        >
          <span>{{ authState.user?.username?.slice(0, 1).toUpperCase() || 'U' }}</span>
          <small>{{ authState.user?.username }}</small>
        </button>
      </div>
    </header>

    <main class="page-main">
      <RouterView />
    </main>
  </div>
</template>
