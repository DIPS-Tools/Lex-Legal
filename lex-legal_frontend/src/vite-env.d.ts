/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_QUERY_API_BASE_URL?: string
  readonly VITE_IMPORT_API_BASE_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
