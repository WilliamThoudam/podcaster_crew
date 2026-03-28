export type ExecuteSqlPayload = {
  success: boolean
  data: Record<string, unknown>[]
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

export type SubResult = {
  sub_question: string
  generated_sql: string
  execute: ExecuteSqlPayload
}

export type QAResponse = {
  answer: string
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

type StreamCallbacks = {
  onDelta?: (deltaText: string) => void
  onProgress?: (event: StreamProgressEvent) => void
}

export type StreamProgressEvent =
  | { type: 'host_plan_started' }
  | { type: 'host_plan_chunk'; chunk: string }
  | { type: 'host_plan_done' }
  | { type: 'planned_sub_questions_started'; total: number }
  | { type: 'planned_sub_questions_chunk'; total: number; chunk: string }
  | { type: 'planned_sub_questions_done'; total: number }
  | { type: 'sub_question_start'; index: number; total: number; sub_question: string }
  | { type: 'sub_question_done'; index: number; total: number; sub_question: string }
  | { type: 'tts_started'; index: number; total: number; sub_question: string }
  | { type: 'tts_label_chunk'; index: number; total: number; sub_question: string; chunk: string }
  | { type: 'tts_generating_chunk'; index: number; total: number; sub_question: string; chunk: string }
  | { type: 'tts_sql_chunk'; index: number; total: number; sub_question: string; chunk: string }
  | { type: 'tts_done'; index: number; total: number; sub_question: string }
  | { type: 'execute_started'; index: number; total: number; sub_question: string }
  | { type: 'execute_label_chunk'; index: number; total: number; sub_question: string; chunk: string }
  | { type: 'execute_generating_chunk'; index: number; total: number; sub_question: string; chunk: string }
  | { type: 'execute_table_chunk'; index: number; total: number; sub_question: string; chunk: string }
  | { type: 'execute_done'; index: number; total: number; sub_question: string }
  | {
      type: 'sub_question_retry'
      index: number
      total: number
      sub_question: string
      reason: 'duplicate_sql_detected'
    }
  | { type: 'summarizing_started' }
  | { type: 'summarizing_chunk'; chunk: string }
  | { type: 'summarizing_done' }

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
  try {
    return JSON.parse(content) as QAResponse
  } catch {
    return { answer: content }
  }
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
    response_format: { type: 'text' },
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
        const progressMatch = delta.match(/^<<PULSECAST_PROGRESS:(.+)>>$/s)
        if (progressMatch) {
          try {
            const event = JSON.parse(progressMatch[1]) as StreamProgressEvent
            callbacks?.onProgress?.(event)
          } catch {
            /* ignore malformed progress marker */
          }
          continue
        }
        assembled += delta
        callbacks?.onDelta?.(delta)
      }
    }
  }

  if (!assembled.trim()) {
    throw new Error('No assistant content received from stream')
  }
  return { answer: assembled }
}
