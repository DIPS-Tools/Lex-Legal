import { createRouter, createWebHistory } from 'vue-router'

import HomeView from './views/HomeView.vue'
import ImportView from './views/ImportView.vue'
import MonitorView from './views/MonitorView.vue'
import QueryView from './views/QueryView.vue'

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
      path: '/legal-compliance',
      alias: ['/legal-query', '/legal-compliance.html'],
      name: 'legal-query',
      component: QueryView,
      meta: { title: 'Legal Query' },
    },
    {
      path: '/import',
      alias: '/import.html',
      name: 'import',
      component: ImportView,
      meta: { title: 'Knowledge Import' },
    },
    {
      path: '/legal-query-monitor',
      alias: '/legal-query-monitor.html',
      name: 'monitor',
      component: MonitorView,
      meta: { title: 'Query Monitor' },
    },
  ],
  scrollBehavior: () => ({ top: 0 }),
})

router.afterEach((route) => {
  document.title = `${String(route.meta.title)} | LexGuard`
})

export default router
