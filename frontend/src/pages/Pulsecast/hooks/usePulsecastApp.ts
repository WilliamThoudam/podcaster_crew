import { useCallback, useEffect, useRef, useState } from 'react'
import type { Dispatch, SetStateAction } from 'react'
import type { NavigateFunction } from 'react-router-dom'
import {
  AGENTS,
  ANALYST_SQL_SNIPPET,
  DURATIONS,
  PODCAST_SCRIPT,
  TOPBAR_BY_SCREEN,
  TOTAL_MS,
} from '../constants'
import { streamPulsecastQa, streamPulsecastResume } from '../../../services/api/pulsecastQa'
import { colorToRgb, cumulativeMsBeforeSegment, fmt, segmentIndexAtElapsed } from '../utils'
import { pathForScreen } from '../../../routes/paths'
import type { AgentState, PodcastRole, QaMessage, Screen } from '../../../types'
import type { StreamProgressEvent } from '../../../services/api/pulsecastQa'

const WAVEFORM_BARS = 80

function newId(): string {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`
}

const QA_INSIGHT_STYLE: Record<PodcastRole, { emoji: string; color: string }> = {
  HOST: { emoji: '🎤', color: 'var(--host)' },
  ANALYST: { emoji: '📊', color: 'var(--analyst)' },
  MARKETING: { emoji: '📣', color: 'var(--marketing)' },
  FINANCE: { emoji: '💰', color: 'var(--finance)' },
  CHALLENGER: { emoji: '⚖️', color: 'var(--challenger)' },
}

/** Sidebar `AGENTS` row index matches PodcastRole order. */
const ROLE_TO_AGENT_INDEX: Record<PodcastRole, number> = {
  HOST: 0,
  ANALYST: 1,
  MARKETING: 2,
  FINANCE: 3,
  CHALLENGER: 4,
}

function agentStatesForThinkingRole(role: PodcastRole | null): AgentState[] {
  return AGENTS.map((_, i) =>
    role != null && ROLE_TO_AGENT_INDEX[role] === i ? 'thinking' : 'idle',
  )
}

type SetThinkingRole = (role: PodcastRole | null) => void

function makePulsecastStreamHandlers(
  typingId: string,
  setQaMessages: Dispatch<SetStateAction<QaMessage[]>>,
  setThinkingRole: SetThinkingRole,
) {
  const markThinking = (role: PodcastRole | null) => {
    setThinkingRole(role)
  }
  let summarizingMsgId: string | null = null
  let typingBubbleCreated = false
  const pushAnalystUpdate = (text: string, sql?: string) => {
    const markdownText = sql ? `${text}\n\n\`\`\`sql\n${sql}\n\`\`\`` : text
    setQaMessages((m) => [
      ...m,
      {
        id: newId(),
        kind: 'agent',
        role: 'ANALYST',
        emoji: QA_INSIGHT_STYLE.ANALYST.emoji,
        color: QA_INSIGHT_STYLE.ANALYST.color,
        text: markdownText,
      },
    ])
  }
  let activeTtsMsgId: string | null = null
  let activeExecMsgId: string | null = null
  let hostPlanMsgId: string | null = null
  let hostPlanText = ''
  let planMsgId: string | null = null
  let planText = ''
  let activeTtsLabel = ''
  let activeTtsGenerating = ''
  let activeTtsSql = ''
  let activeExecLabel = ''
  let activeExecGenerating = ''
  let activeExecTable = ''
  let summarizingText = ''
  let discussionAnalystMsgId: string | null = null
  let discussionAnalystText = ''
  let discussionTurnMsgId: string | null = null
  let discussionTurnText = ''
  let discussionTurnRole: 'MARKETING' | 'FINANCE' | 'CHALLENGER' | null = null
  const upsertAnalystMessage = (id: string, text: string) => {
    setQaMessages((m) => {
      const exists = m.some((msg) => msg.id === id)
      if (!exists) {
        return [
          ...m,
          {
            id,
            kind: 'agent',
            role: 'ANALYST',
            emoji: QA_INSIGHT_STYLE.ANALYST.emoji,
            color: QA_INSIGHT_STYLE.ANALYST.color,
            text,
          },
        ]
      }
      return m.map((msg) => (msg.id === id ? { ...msg, text } : msg))
    })
  }
  const upsertHostMessage = (id: string, text: string) => {
    setQaMessages((m) => {
      const exists = m.some((msg) => msg.id === id)
      if (!exists) {
        return [
          ...m,
          {
            id,
            kind: 'agent',
            role: 'HOST',
            emoji: QA_INSIGHT_STYLE.HOST.emoji,
            color: QA_INSIGHT_STYLE.HOST.color,
            text,
          },
        ]
      }
      return m.map((msg) => (msg.id === id ? { ...msg, text } : msg))
    })
  }
  const upsertRoleMessage = (
    role: 'MARKETING' | 'FINANCE' | 'CHALLENGER',
    id: string,
    text: string,
  ) => {
    const st = QA_INSIGHT_STYLE[role]
    setQaMessages((m) => {
      const exists = m.some((msg) => msg.id === id)
      if (!exists) {
        return [
          ...m,
          {
            id,
            kind: 'agent',
            role,
            emoji: st.emoji,
            color: st.color,
            text,
          },
        ]
      }
      return m.map((msg) => (msg.id === id ? { ...msg, text } : msg))
    })
  }
  const onProgress = (event: StreamProgressEvent) => {
    if (event.type === 'sql_approval_required' || event.type === 'sql_followup_declined') {
      markThinking(null)
      return
    }
    if (event.type === 'host_plan_started') {
      markThinking('HOST')
      hostPlanText = ''
      const id = newId()
      hostPlanMsgId = id
      upsertHostMessage(id, '_Planning…_')
      return
    }
    if (event.type === 'host_plan_chunk') {
      markThinking('HOST')
      if (!hostPlanMsgId) {
        hostPlanMsgId = newId()
      }
      hostPlanText += event.chunk
      upsertHostMessage(hostPlanMsgId, hostPlanText)
      return
    }
    if (event.type === 'host_plan_done') {
      markThinking(null)
      return
    }
    if (event.type === 'planned_sub_questions_started') {
      markThinking('ANALYST')
      planText = ''
      const id = newId()
      planMsgId = id
      upsertAnalystMessage(
        id,
        `To answer this, I will break it down into steps:\n\n_Planning…_`,
      )
      return
    }
    if (event.type === 'planned_sub_questions_chunk') {
      markThinking('ANALYST')
      if (!planMsgId) planMsgId = newId()
      planText += event.chunk
      upsertAnalystMessage(planMsgId, planText)
      return
    }
    if (event.type === 'planned_sub_questions_done') {
      markThinking(null)
      return
    }
    if (event.type === 'sub_question_start') {
      markThinking('ANALYST')
      return
    }
    if (event.type === 'sub_question_done') {
      markThinking(null)
      return
    }
    if (event.type === 'tts_started') {
      markThinking('ANALYST')
      activeTtsLabel = ''
      activeTtsGenerating = ''
      activeTtsSql = ''
      activeTtsMsgId = newId()
      return
    }
    if (event.type === 'tts_label_chunk') {
      markThinking('ANALYST')
      if (!activeTtsMsgId) {
        activeTtsMsgId = newId()
      }
      activeTtsLabel += event.chunk
      upsertAnalystMessage(activeTtsMsgId, activeTtsLabel)
      return
    }
    if (event.type === 'tts_generating_chunk') {
      markThinking('ANALYST')
      if (!activeTtsMsgId) {
        activeTtsMsgId = newId()
      }
      activeTtsGenerating += event.chunk
      upsertAnalystMessage(
        activeTtsMsgId,
        `${activeTtsLabel}\n\n${activeTtsGenerating}`,
      )
      return
    }
    if (event.type === 'tts_sql_chunk') {
      markThinking('ANALYST')
      if (!activeTtsMsgId) {
        activeTtsMsgId = newId()
      }
      activeTtsSql += event.chunk
      const header =
        activeTtsLabel ||
        `Text-to-SQL ${event.index}/${event.total}: ${event.sub_question}`
      upsertAnalystMessage(
        activeTtsMsgId,
        `${header}\n\n\`\`\`sql\n${activeTtsSql}\n\`\`\``,
      )
      return
    }
    if (event.type === 'tts_done') {
      markThinking(null)
      return
    }
    if (event.type === 'execute_started') {
      markThinking('ANALYST')
      activeExecLabel = ''
      activeExecGenerating = ''
      activeExecTable = ''
      activeExecMsgId = newId()
      return
    }
    if (event.type === 'execute_label_chunk') {
      markThinking('ANALYST')
      if (!activeExecMsgId) {
        activeExecMsgId = newId()
      }
      activeExecLabel += event.chunk
      upsertAnalystMessage(activeExecMsgId, activeExecLabel)
      return
    }
    if (event.type === 'execute_generating_chunk') {
      markThinking('ANALYST')
      if (!activeExecMsgId) {
        activeExecMsgId = newId()
      }
      activeExecGenerating += event.chunk
      upsertAnalystMessage(
        activeExecMsgId,
        `${activeExecLabel}\n\n${activeExecGenerating}`,
      )
      return
    }
    if (event.type === 'execute_table_chunk') {
      markThinking('ANALYST')
      if (!activeExecMsgId) {
        activeExecMsgId = newId()
      }
      activeExecTable += event.chunk
      upsertAnalystMessage(
        activeExecMsgId,
        `Result ${event.index}/${event.total}: ${event.sub_question}\n\n${activeExecTable}`,
      )
      return
    }
    if (event.type === 'execute_done') {
      markThinking(null)
      if (activeExecMsgId && activeExecTable) {
        upsertAnalystMessage(
          activeExecMsgId,
          `Result ${event.index}/${event.total}: ${event.sub_question}\n\n${activeExecTable}`,
        )
      }
      return
    }
    if (event.type === 'sub_question_retry') {
      markThinking('ANALYST')
      pushAnalystUpdate(`Retrying ${event.index}/${event.total}: ${event.reason}`)
    }
    if (event.type === 'summarizing_started') {
      markThinking('HOST')
      summarizingText = ''
      summarizingMsgId = newId()
      upsertHostMessage(summarizingMsgId, '_Summarizing…_')
      return
    }
    if (event.type === 'summarizing_chunk') {
      markThinking('HOST')
      if (!summarizingMsgId) {
        summarizingMsgId = newId()
      }
      summarizingText += event.chunk
      upsertHostMessage(summarizingMsgId, summarizingText)
      return
    }
    if (event.type === 'summarizing_done') {
      markThinking(null)
      return
    }
    if (event.type === 'discussion_round_started') {
      markThinking(null)
      pushAnalystUpdate(`_Discussion — round ${event.round}_`)
      return
    }
    if (event.type === 'discussion_analyst_started') {
      markThinking('ANALYST')
      discussionAnalystText = ''
      discussionAnalystMsgId = newId()
      upsertAnalystMessage(
        discussionAnalystMsgId,
        '_Analyst (internal discussion)…_',
      )
      return
    }
    if (event.type === 'discussion_analyst_chunk') {
      markThinking('ANALYST')
      if (!discussionAnalystMsgId) discussionAnalystMsgId = newId()
      discussionAnalystText += event.chunk
      upsertAnalystMessage(discussionAnalystMsgId, discussionAnalystText)
      return
    }
    if (event.type === 'discussion_analyst_done') {
      markThinking(null)
      return
    }
    if (event.type === 'discussion_turn_started') {
      markThinking(event.role)
      discussionTurnText = ''
      discussionTurnMsgId = newId()
      discussionTurnRole = event.role
      upsertRoleMessage(
        event.role,
        discussionTurnMsgId,
        `_Round ${event.round} · ${event.role}…_`,
      )
      return
    }
    if (event.type === 'discussion_turn_chunk') {
      markThinking(event.role)
      if (!discussionTurnMsgId || discussionTurnRole !== event.role) {
        discussionTurnMsgId = newId()
        discussionTurnRole = event.role
      }
      discussionTurnText += event.chunk
      upsertRoleMessage(event.role, discussionTurnMsgId, discussionTurnText)
      return
    }
    if (event.type === 'discussion_turn_done') {
      markThinking(null)
      return
    }
    if (event.type === 'discussion_moderator') {
      const verdict = event.continue_discussion ? 'Continue' : 'Stop'
      pushAnalystUpdate(`_Moderator:_ **${verdict}** — ${event.reason}`)
      return
    }
  }
  const onDelta = (delta: string) => {
    markThinking('HOST')
    const streamMsgId = summarizingMsgId ?? typingId
    if (!typingBubbleCreated) {
      typingBubbleCreated = true
      setQaMessages((m) => {
        const exists = m.some((msg) => msg.id === streamMsgId)
        if (!exists) {
          return [
            ...m,
            {
              id: streamMsgId,
              kind: 'agent',
              role: 'HOST',
              emoji: QA_INSIGHT_STYLE.HOST.emoji,
              color: QA_INSIGHT_STYLE.HOST.color,
              text: delta,
            },
          ]
        }
        return m.map((msg) => (msg.id === streamMsgId ? { ...msg, text: delta } : msg))
      })
    } else {
      setQaMessages((m) =>
        m.map((msg) =>
          msg.id === streamMsgId ? { ...msg, text: `${msg.text}${delta}` } : msg,
        ),
      )
    }
  }
  return {
    onProgress,
    onDelta,
    getSummarizingMsgId: () => summarizingMsgId,
    getTypingBubbleCreated: () => typingBubbleCreated,
  }
}

export function usePulsecastApp(screen: Screen, navigate: NavigateFunction) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [totalElapsed, setTotalElapsed] = useState(0)
  const [currentSegment, setCurrentSegment] = useState(0)
  const [feedItems, setFeedItems] = useState<{ segmentIndex: number; id: string }[]>([])
  const feedSegRef = useRef<number | null>(null)
  const segmentRef = useRef(0)

  const [qaInput, setQaInput] = useState('')
  const qaSessionRef = useRef<string | null>(null)
  const [qaMessages, setQaMessages] = useState<QaMessage[]>([])
  const [qaStreaming, setQaStreaming] = useState(false)

  const [agentStates, setAgentStates] = useState<AgentState[]>(() => AGENTS.map(() => 'idle'))

  const setThinkingRole = useCallback((role: PodcastRole | null) => {
    setAgentStates(agentStatesForThinkingRole(role))
  }, [])

  const [interruptOpen, setInterruptOpen] = useState(false)
  const [interruptDraft, setInterruptDraft] = useState('')

  const [sqlHitlOpen, setSqlHitlOpen] = useState(false)
  const [sqlHitlToken, setSqlHitlToken] = useState<string | null>(null)
  const [sqlHitlProposed, setSqlHitlProposed] = useState('')
  const [sqlHitlEdited, setSqlHitlEdited] = useState('')
  const [sqlHitlRationale, setSqlHitlRationale] = useState<string | null>(null)

  const [toast, setToast] = useState<{ message: string; visible: boolean }>({
    message: '',
    visible: false,
  })

  const [voiceRecording, setVoiceRecording] = useState(false)
  const voiceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const [waveformHeights] = useState(() =>
    Array.from({ length: WAVEFORM_BARS }, () => 10 + Math.random() * 50),
  )

  const topbar = TOPBAR_BY_SCREEN[screen]

  useEffect(() => {
    segmentRef.current = currentSegment
  }, [currentSegment])

  const appendFeedLine = useCallback((segIdx: number) => {
    feedSegRef.current = segIdx
    const id = newId()
    setFeedItems((f) => [...f, { segmentIndex: segIdx, id }])
  }, [])

  const showScreen = useCallback(
    (name: Screen) => {
      navigate(pathForScreen(name))
    },
    [navigate],
  )

  useEffect(() => {
    if (screen !== 'interaction') return
    const id = window.setTimeout(() => {
      setQaMessages((m) => m)
      setAgentStates(AGENTS.map(() => 'idle'))
    }, 0)
    return () => clearTimeout(id)
  }, [screen])

  const showToast = useCallback((message: string) => {
    setToast({ message, visible: true })
    window.setTimeout(() => setToast((t) => ({ ...t, visible: false })), 2800)
  }, [])

  const pausePlayback = useCallback(() => {
    setIsPlaying(false)
    feedSegRef.current = null
    window.speechSynthesis?.cancel()
  }, [])

  const scrubToPercent = useCallback((pct: number) => {
    const clamped = Math.max(0, Math.min(1, pct))
    pausePlayback()
    const ms = Math.floor(clamped * TOTAL_MS)
    setTotalElapsed(ms)
    setCurrentSegment(segmentIndexAtElapsed(ms))
  }, [pausePlayback])

  const startOrTogglePlay = useCallback(() => {
    if (isPlaying) {
      pausePlayback()
      return
    }
    let startSeg = currentSegment
    if (currentSegment >= PODCAST_SCRIPT.length) {
      startSeg = 0
      setCurrentSegment(0)
      setTotalElapsed(0)
      setFeedItems([])
      feedSegRef.current = null
    }
    setIsPlaying(true)
    appendFeedLine(startSeg)
  }, [appendFeedLine, currentSegment, isPlaying, pausePlayback])

  useEffect(() => {
    if (!isPlaying) return
    const iv = window.setInterval(() => {
      setTotalElapsed((e) => {
        const next = Math.min(e + 100, TOTAL_MS)
        if (next >= TOTAL_MS) queueMicrotask(() => setIsPlaying(false))
        return next
      })
    }, 100)
    return () => clearInterval(iv)
  }, [isPlaying])

  useEffect(() => {
    if (!isPlaying) return
    if (currentSegment >= PODCAST_SCRIPT.length) return
    const ms = DURATIONS[currentSegment]
    const fromSeg = currentSegment
    const t = window.setTimeout(() => {
      const next = fromSeg + 1
      setCurrentSegment(next)
      if (next >= PODCAST_SCRIPT.length) {
        setIsPlaying(false)
        window.speechSynthesis?.cancel()
      } else {
        appendFeedLine(next)
      }
    }, ms)
    return () => clearTimeout(t)
  }, [appendFeedLine, isPlaying, currentSegment])

  useEffect(() => {
    if (!isPlaying) return
    if (currentSegment >= PODCAST_SCRIPT.length) return
    const seg = PODCAST_SCRIPT[currentSegment]
    const synth = window.speechSynthesis
    if (!synth) return
    synth.cancel()
    const u = new SpeechSynthesisUtterance(seg.text)
    u.rate = 0.95
    u.pitch =
      seg.role === 'HOST'
        ? 1.1
        : seg.role === 'ANALYST'
          ? 0.9
          : seg.role === 'MARKETING'
            ? 1.05
            : seg.role === 'FINANCE'
              ? 0.85
              : 1.0
    synth.speak(u)
    return () => synth.cancel()
  }, [currentSegment, isPlaying])

  const launchPodcast = useCallback(() => {
    showScreen('player')
    window.setTimeout(() => {
      let s = segmentRef.current
      if (s >= PODCAST_SCRIPT.length) {
        setTotalElapsed(0)
        setFeedItems([])
        feedSegRef.current = null
        setCurrentSegment(0)
        s = 0
      }
      setIsPlaying((playing) => (playing ? playing : true))
      appendFeedLine(s)
    }, 300)
  }, [appendFeedLine, showScreen])

  const openInterrupt = useCallback(() => {
    setInterruptOpen(true)
    if (isPlaying) pausePlayback()
  }, [isPlaying, pausePlayback])

  const closeInterrupt = useCallback(() => {
    setInterruptOpen(false)
    setIsPlaying((p) => {
      if (!p) return true
      return p
    })
  }, [])

  const sendQA = useCallback(
    async (question?: string) => {
      const q = (question ?? qaInput).trim()
      if (!q) return
      setQaInput('')
      setQaMessages((m) => [
        ...m,
        {
          id: newId(),
          kind: 'user',
          role: 'YOU',
          emoji: '👤',
          color: 'var(--accent)',
          text: q,
        },
      ])
      if (!qaSessionRef.current) {
        qaSessionRef.current =
          typeof crypto !== 'undefined' && 'randomUUID' in crypto
            ? crypto.randomUUID()
            : `sess-${Date.now()}`
      }
      setAgentStates(AGENTS.map(() => 'idle'))
      setQaStreaming(true)
      const typingId = newId()
      let stream: ReturnType<typeof makePulsecastStreamHandlers> | null = null
      try {
        stream = makePulsecastStreamHandlers(typingId, setQaMessages, setThinkingRole)
        const conversation = [
          ...qaMessages
            .filter((m) => m.kind === 'user' && m.role === 'YOU' && typeof m.text === 'string')
            .map((m) => ({ role: 'user' as const, content: m.text })),
          { role: 'user' as const, content: q },
        ]
        const result = await streamPulsecastQa(
          {
            question: q,
            session_id: qaSessionRef.current,
            messages: conversation,
          },
          {
            onDelta: stream.onDelta,
            onProgress: stream.onProgress,
          },
        )
        setAgentStates(AGENTS.map(() => 'idle'))
        if (result.kind === 'sql_approval_required') {
          const sid = stream.getSummarizingMsgId()
          const hint =
            '_The analyst suggests an extra query._ Use the dialog to **approve** (run SQL) or **decline** (answer with current data only).'
          if (sid) {
            setQaMessages((m) =>
              m.map((msg) => (msg.id === sid ? { ...msg, text: hint } : msg)),
            )
          } else {
            setQaMessages((m) => [
              ...m,
              {
                id: newId(),
                kind: 'agent',
                role: 'HOST',
                emoji: QA_INSIGHT_STYLE.HOST.emoji,
                color: QA_INSIGHT_STYLE.HOST.color,
                text: hint,
              },
            ])
          }
          setSqlHitlToken(result.resume_token)
          setSqlHitlEdited(result.proposed_sub_question)
          setSqlHitlProposed(result.proposed_sub_question)
          setSqlHitlRationale(result.rationale)
          setSqlHitlOpen(true)
          return
        }
        if (!stream.getTypingBubbleCreated()) {
          setQaMessages((m) => [
            ...m,
            {
              id: newId(),
              kind: 'agent',
              role: 'HOST',
              emoji: QA_INSIGHT_STYLE.HOST.emoji,
              color: QA_INSIGHT_STYLE.HOST.color,
              text: result.answer,
            },
          ])
        }
        const synth = window.speechSynthesis
        if (synth) {
          synth.cancel()
          const u = new SpeechSynthesisUtterance(result.answer)
          u.rate = 0.95
          synth.speak(u)
        }
        window.setTimeout(() => showToast('▶ Podcast resuming from live point…'), 1000)
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e)
        if (stream?.getTypingBubbleCreated()) {
          const streamMsgId = stream.getSummarizingMsgId() ?? typingId
          setQaMessages((m) => m.filter((msgItem) => msgItem.id !== streamMsgId))
        }
        setAgentStates(AGENTS.map(() => 'idle'))
        setQaMessages((m) => [
          ...m,
          {
            id: newId(),
            kind: 'agent',
            role: 'SYSTEM',
            emoji: '⚠️',
            color: 'var(--red)',
            text: msg,
          },
        ])
      } finally {
        setQaStreaming(false)
      }
    },
    [qaInput, qaMessages, setThinkingRole, showToast],
  )

  const submitSqlHitl = useCallback(
    async (approved: boolean) => {
      const token = sqlHitlToken
      if (!token) return
      setSqlHitlOpen(false)
      setQaStreaming(true)
      setAgentStates(AGENTS.map(() => 'idle'))
      const typingId = newId()
      let stream: ReturnType<typeof makePulsecastStreamHandlers> | null = null
      try {
        stream = makePulsecastStreamHandlers(typingId, setQaMessages, setThinkingRole)
        const result = await streamPulsecastResume(
          {
            resume_token: token,
            approved,
            edited_question:
              approved && sqlHitlEdited.trim() ? sqlHitlEdited.trim() : undefined,
            session_id: qaSessionRef.current ?? undefined,
          },
          {
            onDelta: stream.onDelta,
            onProgress: stream.onProgress,
          },
        )
        setAgentStates(AGENTS.map(() => 'idle'))
        setSqlHitlToken(null)
        setSqlHitlProposed('')
        setSqlHitlEdited('')
        setSqlHitlRationale(null)
        if (!stream.getTypingBubbleCreated()) {
          setQaMessages((m) => [
            ...m,
            {
              id: newId(),
              kind: 'agent',
              role: 'HOST',
              emoji: QA_INSIGHT_STYLE.HOST.emoji,
              color: QA_INSIGHT_STYLE.HOST.color,
              text: result.answer,
            },
          ])
        }
        const synth = window.speechSynthesis
        if (synth) {
          synth.cancel()
          const u = new SpeechSynthesisUtterance(result.answer)
          u.rate = 0.95
          synth.speak(u)
        }
        window.setTimeout(() => showToast('▶ Podcast resuming from live point…'), 1000)
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e)
        if (stream?.getTypingBubbleCreated()) {
          const streamMsgId = stream.getSummarizingMsgId() ?? typingId
          setQaMessages((m) => m.filter((msgItem) => msgItem.id !== streamMsgId))
        }
        setSqlHitlToken(null)
        setAgentStates(AGENTS.map(() => 'idle'))
        setQaMessages((m) => [
          ...m,
          {
            id: newId(),
            kind: 'agent',
            role: 'SYSTEM',
            emoji: '⚠️',
            color: 'var(--red)',
            text: msg,
          },
        ])
      } finally {
        setQaStreaming(false)
      }
    },
    [setThinkingRole, showToast, sqlHitlEdited, sqlHitlToken],
  )

  const submitInterrupt = useCallback(() => {
    const q = interruptDraft.trim()
    if (!q) return
    setInterruptDraft('')
    setInterruptOpen(false)
    showScreen('interaction')
    window.setTimeout(() => {
      sendQA(q)
    }, 100)
  }, [interruptDraft, sendQA, showScreen])

  const toggleVoice = useCallback(() => {
    if (voiceRecording) {
      setVoiceRecording(false)
      if (voiceTimerRef.current) clearTimeout(voiceTimerRef.current)
      voiceTimerRef.current = null
      return
    }
    setVoiceRecording(true)
    showToast('🎤 Listening… speak your question')
    voiceTimerRef.current = window.setTimeout(() => {
      setVoiceRecording(false)
      setQaInput('Compare South region with last week')
      showToast('✅ Voice captured')
      voiceTimerRef.current = null
    }, 3000)
  }, [voiceRecording, showToast])

  const skipBack = useCallback(() => {
    const wasPlaying = isPlaying
    const next = Math.max(0, totalElapsed - 10000)
    scrubToPercent(next / TOTAL_MS)
    if (wasPlaying) setIsPlaying(true)
  }, [isPlaying, totalElapsed, scrubToPercent])

  const skipFwd = useCallback(() => {
    const wasPlaying = isPlaying
    const next = Math.min(TOTAL_MS, totalElapsed + 10000)
    scrubToPercent(next / TOTAL_MS)
    if (wasPlaying) setIsPlaying(true)
  }, [isPlaying, totalElapsed, scrubToPercent])

  const onWaveformClick = useCallback(
    (clientX: number, rect: DOMRect) => {
      const pct = (clientX - rect.left) / rect.width
      scrubToPercent(pct)
    },
    [scrubToPercent],
  )

  const onProgressClick = useCallback(
    (clientX: number, rect: DOMRect) => {
      const pct = (clientX - rect.left) / rect.width
      scrubToPercent(pct)
    },
    [scrubToPercent],
  )

  const progressPct = (totalElapsed / TOTAL_MS) * 100
  const playedBarCount = Math.floor((totalElapsed / TOTAL_MS) * waveformHeights.length)

  const displaySegIdx = Math.min(currentSegment, PODCAST_SCRIPT.length - 1)
  const showSpeakerPlaceholder =
    totalElapsed === 0 && !isPlaying && currentSegment === 0

  const speakerSeg = showSpeakerPlaceholder ? null : PODCAST_SCRIPT[displaySegIdx]

  const speakingActive = isPlaying && currentSegment < PODCAST_SCRIPT.length

  return {
    screen,
    showScreen,
    topbarTitle: topbar[0],
    topbarSub: topbar[1],
    isPlaying,
    togglePlay: startOrTogglePlay,
    totalElapsed,
    currentSegment,
    feedItems,
    waveformHeights,
    playedBarCount,
    progressPct,
    speakerSeg,
    showSpeakerPlaceholder,
    speakingActive,
    scrubToPercent,
    skipBack,
    skipFwd,
    onWaveformClick,
    onProgressClick,
    launchPodcast,
    openInterrupt,
    closeInterrupt,
    interruptOpen,
    interruptDraft,
    setInterruptDraft,
    submitInterrupt,
    sqlHitlOpen,
    setSqlHitlOpen,
    sqlHitlProposed,
    sqlHitlEdited,
    setSqlHitlEdited,
    sqlHitlRationale,
    submitSqlHitl,
    qaInput,
    setQaInput,
    sendQA,
    qaMessages,
    qaStreaming,
    agentStates,
    voiceRecording,
    toggleVoice,
    toast,
    showToast,
    colorToRgb,
    fmt,
    cumulativeMsBeforeSegment,
    analystSqlSnippet: ANALYST_SQL_SNIPPET,
  }
}
