<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { login } from '../services/auth'


const route = useRoute()
const router = useRouter()
const username = ref('')
const password = ref('')
const errorMessage = ref('')
const submitting = ref(false)

const requestedPath = computed(() => (
  typeof route.query.redirect === 'string'
    ? route.query.redirect
    : '/legal-compliance'
))

const destinationFor = (isAdmin: boolean) => {
  const resolved = router.resolve(requestedPath.value)
  if (!resolved.matched.length || resolved.path === '/login') {
    return '/legal-compliance'
  }
  if (resolved.meta.requiresAdmin && !isAdmin) {
    return '/legal-compliance'
  }
  return resolved.fullPath
}

const submit = async () => {
  if (!username.value.trim() || !password.value || submitting.value) return
  submitting.value = true
  errorMessage.value = ''
  try {
    const user = await login(username.value, password.value)
    password.value = ''


    //After login, it calls:
    //
    //```typescript
    //router.replace(destinationFor(user.is_admin))
    //```
    //
    //The route guard in `lex-legal_frontend/src/router.ts` calls
    //`restoreAuthentication()`. Because the reactive authentication state is now
    //authenticated, navigation is allowed.
    //
    //Rules are:
    //ordinary users can open `/legal-compliance`;
    //administrators can also open `/import` and `/legal-query-monitor`;
    //an ordinary user requesting an administrator route is redirected to `/legal-compliance`.

    await router.replace(destinationFor(user.is_admin))


  } catch (error) {
    errorMessage.value = error instanceof Error
      ? error.message
      : 'Unable to sign in.'
  } finally {
    submitting.value = false
  }
}
</script>

<template src="../../templates/login.html"></template>
