import { useCallback, useEffect, useRef, useState } from 'react'
import type { NavigateFunction } from 'react-router-dom'
import {
  AGENTS,
  ANALYST_SQL_SNIPPET,
  DURATIONS,
  PODCAST_SCRIPT,
  TOPBAR_BY_SCREEN,
  TOTAL_MS,
} from '../constants'
import { streamPulsecastQa } from '../../../services/api/pulsecastQa'
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

  const [agentStates, setAgentStates] = useState<AgentState[]>(() => AGENTS.map(() => 'idle'))

  const [interruptOpen, setInterruptOpen] = useState(false)
  const [interruptDraft, setInterruptDraft] = useState('')

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
      const typingId = newId()
      let typingBubbleCreated = false
      try {
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
        let planMsgId: string | null = null
        let planText = ''
        let activeTtsSql = ''
        let activeExecTable = ''
        let activeExecStatus = ''
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
        const onProgress = (event: StreamProgressEvent) => {
          if (event.type === 'planned_sub_questions_started') {
            planText = ''
            const id = newId()
            planMsgId = id
            upsertAnalystMessage(id, `Planned sub-questions:\n\n_Planning…_`)
            return
          }
          if (event.type === 'planned_sub_questions_chunk') {
            if (!planMsgId) planMsgId = newId()
            planText += event.chunk
            upsertAnalystMessage(planMsgId, planText)
            return
          }
          if (event.type === 'planned_sub_questions_done') {
            return
          }
          if (event.type === 'sub_question_start') return
          if (event.type === 'sub_question_done') {
            return
          }
          if (event.type === 'tts_started') {
            activeTtsSql = ''
            const id = newId()
            activeTtsMsgId = id
            upsertAnalystMessage(id, `Text-to-SQL ${event.index}/${event.total}: ${event.sub_question}\n\n_Generating…_`)
            return
          }
          if (event.type === 'tts_sql_chunk') {
            if (!activeTtsMsgId) {
              activeTtsMsgId = newId()
            }
            activeTtsSql += event.chunk
            upsertAnalystMessage(
              activeTtsMsgId,
              `Text-to-SQL ${event.index}/${event.total}: ${event.sub_question}\n\n\`\`\`sql\n${activeTtsSql}\n\`\`\``,
            )
            return
          }
          if (event.type === 'tts_done') {
            return
          }
          if (event.type === 'execute_started') {
            activeExecTable = ''
            activeExecStatus = ''
            const id = newId()
            activeExecMsgId = id
            upsertAnalystMessage(id, `Executing ${event.index}/${event.total}: ${event.sub_question}\n\n`)
            return
          }
          if (event.type === 'execute_status_chunk') {
            if (!activeExecMsgId) {
              activeExecMsgId = newId()
            }
            activeExecStatus += event.chunk
            upsertAnalystMessage(
              activeExecMsgId,
              `Executing ${event.index}/${event.total}: ${event.sub_question}\n\n${activeExecStatus}`,
            )
            return
          }
          if (event.type === 'execute_table_chunk') {
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
            const rc = event.row_count ?? 0
            if (activeExecMsgId && activeExecTable) {
              upsertAnalystMessage(
                activeExecMsgId,
                `Result ${event.index}/${event.total}: ${event.sub_question} (rows: ${rc})\n\n${activeExecTable}`,
              )
            }
            return
          }
          if (event.type === 'sub_question_retry') {
            pushAnalystUpdate(`Retrying ${event.index}/${event.total}: ${event.reason}`)
          }
        }
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
            onDelta: (delta) => {
              if (!typingBubbleCreated) {
                typingBubbleCreated = true
                setQaMessages((m) => [
                  ...m,
                  {
                    id: typingId,
                    kind: 'agent',
                    role: 'HOST',
                    emoji: QA_INSIGHT_STYLE.HOST.emoji,
                    color: QA_INSIGHT_STYLE.HOST.color,
                    text: delta,
                  },
                ])
              } else {
                setQaMessages((m) =>
                  m.map((msg) => (msg.id === typingId ? { ...msg, text: `${msg.text}${delta}` } : msg)),
                )
              }
            },
            onProgress,
          },
        )
        setAgentStates(AGENTS.map(() => 'idle'))
        if (!typingBubbleCreated) {
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
        if (typingBubbleCreated) {
          setQaMessages((m) => m.filter((msgItem) => msgItem.id !== typingId))
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
      }
    },
    [qaInput, qaMessages, showToast],
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
    qaInput,
    setQaInput,
    sendQA,
    qaMessages,
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
