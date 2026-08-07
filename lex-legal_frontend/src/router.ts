import { createRouter, createWebHistory } from 'vue-router'

import HomeView from './views/HomeView.vue'
import ImportView from './views/ImportView.vue'
import MonitorView from './views/MonitorView.vue'
import QueryView from './views/QueryView.vue'
import LoginView from './views/LoginView.vue'
import { authState, restoreAuthentication } from './services/auth'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      alias: '/index.html',
      name: 'home',
      component: HomeView,
      meta: { title: 'Legal Compliance Assistant' },
    },
    {
      path: '/login',
      name: 'login',
      component: LoginView,
      meta: { title: 'Sign In' },
    },
    {
      path: '/legal-compliance',
      alias: ['/legal-query', '/legal-compliance.html'],
      name: 'legal-query',
      component: QueryView,
      meta: { title: 'Legal Query', requiresAuth: true },
    },
    {
      path: '/import',
      alias: '/import.html',
      name: 'import',
      component: ImportView,
      meta: {
        title: 'Knowledge Import',
        requiresAuth: true,
        requiresAdmin: true,
      },
    },
    {
      path: '/legal-query-monitor',
      alias: '/legal-query-monitor.html',
      name: 'monitor',
      component: MonitorView,
      meta: {
        title: 'Query Monitor',
        requiresAuth: true,
        requiresAdmin: true,
      },
    },
  ],
  scrollBehavior: () => ({ top: 0 }),
})

router.beforeEach(async (route) => {
  const requiresAuth = Boolean(route.meta.requiresAuth)
  const isLogin = route.name === 'login'
  if (!requiresAuth && !isLogin) return true

  await restoreAuthentication()
  if (isLogin && authState.status === 'authenticated') {
    return authState.user?.is_admin ? '/' : '/legal-compliance'
  }
  if (requiresAuth && authState.status !== 'authenticated') {
    return {
      name: 'login',
      query: { redirect: route.fullPath },
    }
  }
  if (route.meta.requiresAdmin && !authState.user?.is_admin) {
    return '/legal-compliance'
  }
  return true
})

router.afterEach((route) => {
  document.title = `${String(route.meta.title)} | LexGuard`
})

export default router
