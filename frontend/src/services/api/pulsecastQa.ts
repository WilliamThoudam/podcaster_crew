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
  session_id?: string
  messages?: Array<{ role: 'user'; content: string }>
}

type OpenAIMessage = {
  role: 'user' | 'assistant' | 'system'
  content: string
}

type OpenAIChatRequest = {
  model: string
  messages: OpenAIMessage[]
  stream: boolean
  temperature?: number
  response_format?: { type: 'json_object' | 'text' }
  user?: string
}

type OpenAIChoice = {
  index: number
  message: OpenAIMessage
  finish_reason?: string | null
}

type OpenAIChatResponse = {
  id: string
  object: 'chat.completion'
  created: number
  model: string
  choices: OpenAIChoice[]
}

type OpenAIChunkChoice = {
  index: number
  delta?: { role?: 'assistant'; content?: string }
  finish_reason?: string | null
}

type OpenAIChunk = {
  id: string
  object: 'chat.completion.chunk'
  created: number
  model: string
  choices: OpenAIChunkChoice[]
}

type StreamEventPayload =
  | { event: 'agent_status'; agent: 'HOST' | 'ANALYST' | 'MARKETING' | 'FINANCE' | 'CHALLENGER'; state: 'thinking' | 'idle' | 'active' }
  | { event: 'agent_message_start'; agent: 'HOST' | 'ANALYST' | 'MARKETING' | 'FINANCE' | 'CHALLENGER' }
  | {
      event: 'agent_text_delta'
      agent: 'HOST' | 'ANALYST' | 'MARKETING' | 'FINANCE' | 'CHALLENGER'
      delta: string
    }
  | {
      event: 'agent_message_done'
      agent: 'HOST' | 'ANALYST' | 'MARKETING' | 'FINANCE' | 'CHALLENGER'
      state: 'idle' | 'thinking' | 'active'
      phase?: string
      detail?: string | null
    }
  | { event: 'sql_status'; stage: string; generated_sql?: string }
  | { event: 'final_payload'; payload: QAResponse }
  | { event: 'error'; message: string }

type StreamCallbacks = {
  onDelta?: (deltaText: string) => void
  onAgentStatus?: (
    agent: 'HOST' | 'ANALYST' | 'MARKETING' | 'FINANCE' | 'CHALLENGER',
    state: 'thinking' | 'idle' | 'active',
    meta?: { phase?: string; detail?: string | null },
  ) => void
  onAgentMessageStart?: (agent: 'HOST' | 'ANALYST' | 'MARKETING' | 'FINANCE' | 'CHALLENGER') => void
  onAgentTextDelta?: (
    agent: 'HOST' | 'ANALYST' | 'MARKETING' | 'FINANCE' | 'CHALLENGER',
    delta: string,
  ) => void
  onAgentMessageDone?: (
    agent: 'HOST' | 'ANALYST' | 'MARKETING' | 'FINANCE' | 'CHALLENGER',
    meta?: { phase?: string; detail?: string | null; state?: 'idle' | 'thinking' | 'active' },
  ) => void
  onFinalPayload?: (payload: QAResponse) => void
  onStreamError?: (message: string) => void
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
  const req: OpenAIChatRequest = {
    model: 'pulsecast-qa',
    stream: false,
    messages:
      body.messages && body.messages.length > 0
        ? body.messages
        : [{ role: 'user', content: body.question }],
    response_format: { type: 'json_object' },
    user: body.session_id,
  }
  const res = await fetch(`${apiBase()}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })

  if (!res.ok) {
    let msg = res.statusText
    try {
      const j = (await res.json()) as { error?: { message?: unknown }; detail?: unknown }
      if (j.error?.message !== undefined) msg = parseDetail(j.error.message)
      else if (j.detail !== undefined) msg = parseDetail(j.detail)
    } catch {
      /* ignore */
    }
    throw new Error(msg)
  }

  const completion = (await res.json()) as OpenAIChatResponse
  const content = completion.choices?.[0]?.message?.content
  if (!content) throw new Error('Missing assistant content in completion response')
  return JSON.parse(content) as QAResponse
}

function parseSseEvents(chunk: string): string[] {
  return chunk
    .split('\n\n')
    .map((block) => block.trim())
    .filter(Boolean)
    .filter((block) => block.startsWith('data:'))
    .map((block) => block.slice('data:'.length).trim())
}

export async function streamPulsecastQa(
  body: QARequestBody,
  callbacks?: StreamCallbacks,
): Promise<QAResponse> {
  const req: OpenAIChatRequest = {
    model: 'pulsecast-qa',
    stream: true,
    messages:
      body.messages && body.messages.length > 0
        ? body.messages
        : [{ role: 'user', content: body.question }],
    response_format: { type: 'json_object' },
    user: body.session_id,
  }
  const res = await fetch(`${apiBase()}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) {
    let msg = res.statusText
    try {
      const j = (await res.json()) as { error?: { message?: unknown }; detail?: unknown }
      if (j.error?.message !== undefined) msg = parseDetail(j.error.message)
      else if (j.detail !== undefined) msg = parseDetail(j.detail)
    } catch {
      /* ignore */
    }
    throw new Error(msg)
  }
  if (!res.body) throw new Error('Streaming response body is unavailable')

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let rawBuffer = ''
  let assembled = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    rawBuffer += decoder.decode(value, { stream: true })

    const boundary = rawBuffer.lastIndexOf('\n\n')
    if (boundary === -1) continue

    const ready = rawBuffer.slice(0, boundary + 2)
    rawBuffer = rawBuffer.slice(boundary + 2)

    for (const dataLine of parseSseEvents(ready)) {
      if (dataLine === '[DONE]') continue
      const chunk = JSON.parse(dataLine) as OpenAIChunk
      const delta = chunk.choices?.[0]?.delta?.content
      if (delta) {
        // Realtime typed event transport in delta.content
        let handledAsEvent = false
        try {
          const evt = JSON.parse(delta) as StreamEventPayload
          if (evt && typeof evt === 'object' && 'event' in evt) {
            handledAsEvent = true
            if (evt.event === 'agent_status') {
              callbacks?.onAgentStatus?.(evt.agent, evt.state)
            } else if (evt.event === 'agent_message_start') {
              callbacks?.onAgentMessageStart?.(evt.agent)
            } else if (evt.event === 'agent_text_delta') {
              callbacks?.onAgentTextDelta?.(evt.agent, evt.delta)
            } else if (evt.event === 'agent_message_done') {
              callbacks?.onAgentStatus?.(evt.agent, evt.state, {
                phase: evt.phase,
                detail: evt.detail ?? null,
              })
              callbacks?.onAgentMessageDone?.(evt.agent, {
                phase: evt.phase,
                detail: evt.detail ?? null,
                state: evt.state,
              })
            } else if (evt.event === 'sql_status') {
              // no-op: hook can still infer progress from agent updates
            } else if (evt.event === 'final_payload') {
              callbacks?.onFinalPayload?.(evt.payload)
              assembled = JSON.stringify(evt.payload)
            } else if (evt.event === 'error') {
              callbacks?.onStreamError?.(evt.message)
            }
          }
        } catch {
          handledAsEvent = false
        }
        if (!handledAsEvent) {
          assembled += delta
          callbacks?.onDelta?.(delta)
        }
      }
    }
  }

  if (!assembled.trim()) {
    throw new Error('No assistant content received from stream')
  }
  return JSON.parse(assembled) as QAResponse
}
