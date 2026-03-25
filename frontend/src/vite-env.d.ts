/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_PULSECAST_API_URL?: string
  readonly VITE_PULSECAST_USER_ID?: string
  readonly VITE_PULSECAST_USER_DB_ID?: string
  readonly VITE_PULSECAST_DB_TYPE?: string
  readonly VITE_PULSECAST_SCHEMA_NAME?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
