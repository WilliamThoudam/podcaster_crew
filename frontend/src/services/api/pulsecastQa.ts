export type ExecuteSqlField = {
  name: string
  type?: string | null
  scale?: number | null
  nullable?: boolean | null
}

export type ExecuteSqlPayload = {
  success: boolean
  data: Record<string, unknown>[]
  rowCount?: number | null
  fields: ExecuteSqlField[]
  executionTime?: number | null
  query?: string | null
  originalQuery?: string | null
  limited?: boolean | null
  maxRecords?: number | null
  note?: string | null
}

export type PulsecastAgentId = 'host' | 'analyst' | 'marketing' | 'finance' | 'challenger'

export type AgentPipelineStep = {
  id: PulsecastAgentId
  status: 'completed' | 'skipped'
  phase: string
  detail?: string | null
}

export type AgentInsight = {
  role: 'HOST' | 'ANALYST' | 'MARKETING' | 'FINANCE' | 'CHALLENGER'
  text: string
}

export type QAResponse = {
  generated_sql: string
  answer: string
  execute: ExecuteSqlPayload
  text_to_sql_error?: string | null
  pipeline?: AgentPipelineStep[]
  agent_messages?: AgentInsight[]
}

export type QARequestBody = {
  question: string
  user_id?: number
  user_db_id?: number
  db_type?: string
  schema_name?: string
  session_id?: string
  model?: string
  max_nodes?: string
  is_retry?: boolean
}

function apiBase(): string {
  const b = import.meta.env.VITE_PULSECAST_API_URL
  return (typeof b === 'string' && b.length > 0 ? b : 'http://localhost:8000').replace(/\/$/, '')
}

function parseDetail(detail: unknown): string {
  if (detail == null) return 'Request failed'
  if (typeof detail === 'string') return detail
  try {
    return JSON.stringify(detail)
  } catch {
    return String(detail)
  }
}

export async function postPulsecastQa(body: QARequestBody): Promise<QAResponse> {
  const optional: QARequestBody = { ...body }
  const uid = import.meta.env.VITE_PULSECAST_USER_ID
  const udb = import.meta.env.VITE_PULSECAST_USER_DB_ID
  if (optional.user_id === undefined && uid !== undefined && uid !== '') {
    optional.user_id = Number(uid)
  }
  if (optional.user_db_id === undefined && udb !== undefined && udb !== '') {
    optional.user_db_id = Number(udb)
  }
  const dbType = import.meta.env.VITE_PULSECAST_DB_TYPE
  if (optional.db_type === undefined && typeof dbType === 'string' && dbType) {
    optional.db_type = dbType
  }
  const schema = import.meta.env.VITE_PULSECAST_SCHEMA_NAME
  if (optional.schema_name === undefined && typeof schema === 'string' && schema) {
    optional.schema_name = schema
  }

  const res = await fetch(`${apiBase()}/api/v1/qa`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(optional),
  })

  if (!res.ok) {
    let msg = res.statusText
    try {
      const j = (await res.json()) as { detail?: unknown }
      if (j.detail !== undefined) msg = parseDetail(j.detail)
    } catch {
      /* ignore */
    }
    throw new Error(msg)
  }

  return res.json() as Promise<QAResponse>
}
