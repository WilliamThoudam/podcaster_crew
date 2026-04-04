export type ExecuteSqlPayload = {
  success: boolean
  data: Record<string, unknown>[]
}

export type PulsecastAgentId =
  | 'host'
  | 'analyst'
  | 'marketing'
  | 'finance'
  | 'forecaster'
  | 'web_crawler'
  | 'challenger'

export type AgentPipelineStep = {
  id: PulsecastAgentId
  status: 'completed' | 'skipped'
  phase: string
  detail?: string | null
}

export type AgentInsight = {
  role: 'HOST' | 'ANALYST' | 'MARKETING' | 'FINANCE' | 'FORECASTER' | 'WEB_CRAWLER' | 'CHALLENGER'
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

/** Optional controls for streaming QA/resume requests. */
export type PulsecastStreamOptions = {
  signal?: AbortSignal
  /** Called with job id from response headers before the SSE body is consumed (for pause/resume). */
  onStreamJobId?: (jobId: string) => void
}

export type SqlHitlPauseKind = 'challenger_followup' | 'duplicate_sub_question' | 'web_search'

export type StreamQaOutcome =
  | { kind: 'aborted' }
  | { kind: 'complete'; answer: string }
  | {
      kind: 'sql_approval_required'
      resume_token: string
      proposed_sub_question: string
      rationale: string | null
      pause_kind?: SqlHitlPauseKind
      /** Present when pause_kind is web_search and the plan has multiple Serper steps */
      web_search_step_index?: number
      web_search_total_steps?: number
    }
  | {
      kind: 'discussion_approval_required'
      resume_token: string
      stage: 'pre' | 'mid'
      requested_depth?: 'linear' | 'moderated'
      round_index: number
      max_rounds: number
      focus_for_next_round?: string | null
      rationale: string | null
    }

export type PulsecastResumeRequestBody = {
  resume_token: string
  approved: boolean
  edited_question?: string
  /** Merge into paused discussion question (skips full /refine replan). */
  discussion_refinement?: string
  session_id?: string
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
  | { type: 'aggregation_started'; index: number; total: number; sub_question: string; stage?: string }
  | {
      type: 'aggregation_done'
      index: number
      total: number
      sub_question: string
      stage?: string
      action: 'pass_through' | 'rewrite' | 'skipped'
      was_rewritten: boolean
      sql_changed: boolean
      confidence?: number | null
      reason?: string | null
    }
  | { type: 'query_rewritten'; index: number; total: number; sub_question: string; reason: string }
  | {
      type: 'rowcount_exceeded'
      index: number
      total: number
      sub_question: string
      row_count: number
      threshold: number
    }
  | {
      type: 'aggregation_sql_chunk'
      index: number
      total: number
      sub_question: string
      chunk: string
      stage?: string
    }
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
  | { type: 'web_search_results_started'; query: string }
  | { type: 'web_search_results_label_chunk'; query: string; chunk: string }
  | { type: 'web_search_results_table_chunk'; query: string; chunk: string }
  | { type: 'web_search_results_done'; query: string }
  | {
      type: 'sub_question_retry'
      index: number
      total: number
      sub_question: string
      reason: 'duplicate_sql_detected'
      /** Short user-facing sentence from the API */
      message?: string
    }
  | { type: 'summarizing_started' }
  | { type: 'summarizing_chunk'; chunk: string }
  | { type: 'summarizing_done' }
  | { type: 'discussion_round_started'; round: number }
  | { type: 'discussion_analyst_started' }
  | { type: 'discussion_analyst_chunk'; chunk: string }
  | { type: 'discussion_analyst_done' }
  | {
      type: 'discussion_turn_started'
      role: 'MARKETING' | 'FINANCE' | 'FORECASTER' | 'WEB_CRAWLER' | 'CHALLENGER'
      round: number
    }
  | {
      type: 'discussion_turn_chunk'
      role: 'MARKETING' | 'FINANCE' | 'FORECASTER' | 'WEB_CRAWLER' | 'CHALLENGER'
      round: number
      chunk: string
    }
  | {
      type: 'discussion_turn_done'
      role: 'MARKETING' | 'FINANCE' | 'FORECASTER' | 'WEB_CRAWLER' | 'CHALLENGER'
      round: number
    }
  | { type: 'discussion_moderator'; continue_discussion: boolean; reason: string }
  | {
      type: 'sql_approval_required'
      resume_token: string
      proposed_sub_question: string
      rationale?: string | null
      pause_kind?: SqlHitlPauseKind
    }
  | {
      type: 'web_search_approval_required'
      resume_token: string
      proposed_search_query: string
      rationale?: string | null
      pause_kind?: 'web_search'
      web_search_step_index?: number
      web_search_total_steps?: number
    }
  | {
      type: 'discussion_approval_required'
      resume_token: string
      pause_kind: 'discussion'
      stage: 'pre' | 'mid'
      requested_depth?: 'linear' | 'moderated'
      round_index: number
      max_rounds: number
      focus_for_next_round?: string | null
      rationale?: string | null
    }
  | { type: 'sql_followup_declined' }
  | { type: 'web_search_declined' }
  | { type: 'duplicate_sub_question_skipped' }

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

type ConsumeSseResult =
  | { outcome: 'aborted' }
  | { outcome: 'complete'; assembled: string }
  | {
      outcome: 'sql_approval_required'
      resume_token: string
      proposed_sub_question: string
      rationale: string | null
      pause_kind: SqlHitlPauseKind
      web_search_step_index?: number
      web_search_total_steps?: number
    }
  | {
      outcome: 'discussion_approval_required'
      resume_token: string
      stage: 'pre' | 'mid'
      requested_depth?: 'linear' | 'moderated'
      round_index: number
      max_rounds: number
      focus_for_next_round?: string | null
      rationale: string | null
    }

function isAbortError(e: unknown): boolean {
  return (
    (e instanceof DOMException && e.name === 'AbortError') ||
    (typeof e === 'object' &&
      e !== null &&
      'name' in e &&
      (e as { name: string }).name === 'AbortError')
  )
}

async function consumeSseChatStream(
  res: Response,
  callbacks?: StreamCallbacks,
  streamOptions?: PulsecastStreamOptions,
): Promise<ConsumeSseResult> {
  if (!res.body) throw new Error('Streaming response body is unavailable')

  const signal = streamOptions?.signal
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let rawBuffer = ''
  let assembled = ''
  let sqlApproval:
    | {
        resume_token: string
        proposed_sub_question: string
        rationale: string | null
        pause_kind: SqlHitlPauseKind
        web_search_step_index?: number
        web_search_total_steps?: number
      }
    | undefined
  let discussionApproval:
    | {
        resume_token: string
        stage: 'pre' | 'mid'
        requested_depth?: 'linear' | 'moderated'
        round_index: number
        max_rounds: number
        focus_for_next_round?: string | null
        rationale: string | null
      }
    | undefined

  try {
    while (true) {
      if (signal?.aborted) {
        await reader.cancel().catch(() => {})
        return { outcome: 'aborted' }
      }
      let chunk: ReadableStreamReadResult<Uint8Array>
      try {
        chunk = await reader.read()
      } catch (e) {
        if (isAbortError(e)) return { outcome: 'aborted' }
        throw e
      }
      const { done, value } = chunk
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
            if (event.type === 'sql_approval_required') {
              sqlApproval = {
                resume_token: event.resume_token,
                proposed_sub_question: event.proposed_sub_question,
                rationale: event.rationale ?? null,
                pause_kind: event.pause_kind ?? 'challenger_followup',
              }
            }
            if (event.type === 'web_search_approval_required') {
              sqlApproval = {
                resume_token: event.resume_token,
                proposed_sub_question: event.proposed_search_query,
                rationale: event.rationale ?? null,
                pause_kind: 'web_search',
                web_search_step_index: event.web_search_step_index,
                web_search_total_steps: event.web_search_total_steps,
              }
            }
            if (event.type === 'discussion_approval_required') {
              discussionApproval = {
                resume_token: event.resume_token,
                stage: event.stage,
                requested_depth: event.requested_depth,
                round_index: event.round_index,
                max_rounds: event.max_rounds,
                focus_for_next_round: event.focus_for_next_round ?? null,
                rationale: event.rationale ?? null,
              }
            }
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
  } catch (e) {
    if (isAbortError(e)) return { outcome: 'aborted' }
    throw e
  }

  if (!assembled.trim()) {
    if (sqlApproval) {
      return {
        outcome: 'sql_approval_required',
        resume_token: sqlApproval.resume_token,
        proposed_sub_question: sqlApproval.proposed_sub_question,
        rationale: sqlApproval.rationale,
        pause_kind: sqlApproval.pause_kind,
        web_search_step_index: sqlApproval.web_search_step_index,
        web_search_total_steps: sqlApproval.web_search_total_steps,
      }
    }
    if (discussionApproval) {
      return {
        outcome: 'discussion_approval_required',
        resume_token: discussionApproval.resume_token,
        stage: discussionApproval.stage,
        requested_depth: discussionApproval.requested_depth,
        round_index: discussionApproval.round_index,
        max_rounds: discussionApproval.max_rounds,
        focus_for_next_round: discussionApproval.focus_for_next_round,
        rationale: discussionApproval.rationale,
      }
    }
    throw new Error('No assistant content received from stream')
  }
  return { outcome: 'complete', assembled }
}

export async function pulsecastStreamControl(body: {
  job_id: string
  paused: boolean
  user?: string
}): Promise<void> {
  const res = await fetch(`${apiBase()}/v1/chat/completions/stream-control`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      job_id: body.job_id,
      paused: body.paused,
      user: body.user ?? null,
    }),
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
}

export type PulsecastRefineRequestBody = {
  session_id: string
  refinement: string
}

export async function streamPulsecastRefine(
  body: PulsecastRefineRequestBody,
  callbacks?: StreamCallbacks,
  streamOptions?: PulsecastStreamOptions,
): Promise<StreamQaOutcome> {
  const req = {
    model: 'pulsecast-qa',
    stream: true,
    session_id: body.session_id,
    refinement: body.refinement,
  }
  let res: Response
  try {
    res = await fetch(`${apiBase()}/v1/chat/completions/refine`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
      signal: streamOptions?.signal,
    })
  } catch (e) {
    if (isAbortError(e)) return { kind: 'aborted' }
    throw e
  }
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

  const jobHeader = res.headers.get('X-Pulsecast-Stream-Job-Id')
  if (jobHeader) streamOptions?.onStreamJobId?.(jobHeader)

  const raw = await consumeSseChatStream(res, callbacks, streamOptions)
  if (raw.outcome === 'aborted') return { kind: 'aborted' }
  if (raw.outcome === 'sql_approval_required') {
    return {
      kind: 'sql_approval_required',
      resume_token: raw.resume_token,
      proposed_sub_question: raw.proposed_sub_question,
      rationale: raw.rationale,
      pause_kind: raw.pause_kind,
      web_search_step_index: raw.web_search_step_index,
      web_search_total_steps: raw.web_search_total_steps,
    }
  }
  if (raw.outcome === 'discussion_approval_required') {
    return {
      kind: 'discussion_approval_required',
      resume_token: raw.resume_token,
      stage: raw.stage,
      requested_depth: raw.requested_depth,
      round_index: raw.round_index,
      max_rounds: raw.max_rounds,
      focus_for_next_round: raw.focus_for_next_round,
      rationale: raw.rationale,
    }
  }
  return { kind: 'complete', answer: raw.assembled }
}

export async function streamPulsecastQa(
  body: QARequestBody,
  callbacks?: StreamCallbacks,
  streamOptions?: PulsecastStreamOptions,
): Promise<StreamQaOutcome> {
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
  let res: Response
  try {
    res = await fetch(`${apiBase()}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
      signal: streamOptions?.signal,
    })
  } catch (e) {
    if (isAbortError(e)) return { kind: 'aborted' }
    throw e
  }
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

  const jobHeader = res.headers.get('X-Pulsecast-Stream-Job-Id')
  if (jobHeader) streamOptions?.onStreamJobId?.(jobHeader)

  const raw = await consumeSseChatStream(res, callbacks, streamOptions)
  if (raw.outcome === 'aborted') return { kind: 'aborted' }
  if (raw.outcome === 'sql_approval_required') {
    return {
      kind: 'sql_approval_required',
      resume_token: raw.resume_token,
      proposed_sub_question: raw.proposed_sub_question,
      rationale: raw.rationale,
      pause_kind: raw.pause_kind,
      web_search_step_index: raw.web_search_step_index,
      web_search_total_steps: raw.web_search_total_steps,
    }
  }
  if (raw.outcome === 'discussion_approval_required') {
    return {
      kind: 'discussion_approval_required',
      resume_token: raw.resume_token,
      stage: raw.stage,
      requested_depth: raw.requested_depth,
      round_index: raw.round_index,
      max_rounds: raw.max_rounds,
      focus_for_next_round: raw.focus_for_next_round,
      rationale: raw.rationale,
    }
  }
  return { kind: 'complete', answer: raw.assembled }
}

export async function streamPulsecastResume(
  body: PulsecastResumeRequestBody,
  callbacks?: StreamCallbacks,
  streamOptions?: PulsecastStreamOptions,
): Promise<StreamQaOutcome> {
  const req = {
    model: 'pulsecast-qa',
    stream: true,
    resume_token: body.resume_token,
    approved: body.approved,
    edited_question: body.edited_question,
    discussion_refinement: body.discussion_refinement,
    user: body.session_id,
  }
  let res: Response
  try {
    res = await fetch(`${apiBase()}/v1/chat/completions/resume`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
      signal: streamOptions?.signal,
    })
  } catch (e) {
    if (isAbortError(e)) return { kind: 'aborted' }
    throw e
  }
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

  const jobHeader = res.headers.get('X-Pulsecast-Stream-Job-Id')
  if (jobHeader) streamOptions?.onStreamJobId?.(jobHeader)

  const raw = await consumeSseChatStream(res, callbacks, streamOptions)
  if (raw.outcome === 'aborted') return { kind: 'aborted' }
  if (raw.outcome === 'sql_approval_required') {
    return {
      kind: 'sql_approval_required',
      resume_token: raw.resume_token,
      proposed_sub_question: raw.proposed_sub_question,
      rationale: raw.rationale,
      pause_kind: raw.pause_kind,
      web_search_step_index: raw.web_search_step_index,
      web_search_total_steps: raw.web_search_total_steps,
    }
  }
  if (raw.outcome === 'discussion_approval_required') {
    return {
      kind: 'discussion_approval_required',
      resume_token: raw.resume_token,
      stage: raw.stage,
      requested_depth: raw.requested_depth,
      round_index: raw.round_index,
      max_rounds: raw.max_rounds,
      focus_for_next_round: raw.focus_for_next_round,
      rationale: raw.rationale,
    }
  }
  return { kind: 'complete', answer: raw.assembled }
}
