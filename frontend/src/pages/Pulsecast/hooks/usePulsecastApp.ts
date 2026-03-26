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
import type { AgentInsight } from '../../../services/api/pulsecastQa'

const WAVEFORM_BARS = 80

const INTERACTION_WELCOME: QaMessage = {
  id: 'sys-welcome',
  kind: 'system',
  role: 'SYSTEM',
  emoji: '🎙️',
  color: 'var(--accent)',
  text: 'Podcast paused for your session. Ask any question about the data and the agents will respond with live SQL analysis.',
}

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

const ROLE_TO_AGENT_INDEX: Record<PodcastRole, number> = {
  HOST: 0,
  ANALYST: 1,
  MARKETING: 2,
  FINANCE: 3,
  CHALLENGER: 4,
}

const ROLE_TO_UI_META: Record<PodcastRole, { emoji: string; color: string }> = QA_INSIGHT_STYLE

function insightToMessage(ins: AgentInsight, generatedSql: string): QaMessage {
  const role = ins.role in QA_INSIGHT_STYLE ? ins.role : 'ANALYST'
  const style = QA_INSIGHT_STYLE[role as PodcastRole]
  return {
    id: newId(),
    kind: 'agent',
    role,
    emoji: style.emoji,
    color: style.color,
    text: ins.text,
    ...(role === 'ANALYST' ? { sql: generatedSql } : {}),
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
  const streamingMsgIdsRef = useRef<Partial<Record<PodcastRole, string>>>({})
  const [qaMessages, setQaMessages] = useState<QaMessage[]>([])

  const [agentStates, setAgentStates] = useState<AgentState[]>(() => AGENTS.map(() => 'idle'))
  const [sqlLog, setSqlLog] = useState('// Waiting for query…')

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
      setQaMessages((m) => (m.length === 0 ? [INTERACTION_WELCOME] : m))
      setAgentStates(AGENTS.map(() => 'idle'))
      setSqlLog('// Waiting for query…')
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
      setAgentStates(AGENTS.map((_, i) => (i === 1 ? 'thinking' : 'idle')))
      setSqlLog('// Streaming OpenAI completion…')
      try {
        let streamedChars = 0
        let streamedAgentMessages = false
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
              streamedChars += delta.length
              if (streamedChars % 256 < delta.length) {
                setSqlLog(`// Streaming OpenAI completion… ${streamedChars} chars`)
              }
            },
            onAgentStatus: (agent, state) => {
              const idx = ROLE_TO_AGENT_INDEX[agent]
              setAgentStates((prev) => {
                const next: AgentState[] = [...prev]
                next[idx] = state
                return next
              })
            },
            onAgentMessageStart: (agent) => {
              streamedAgentMessages = true
              const id = newId()
              streamingMsgIdsRef.current[agent] = id
              const meta = ROLE_TO_UI_META[agent]
              setQaMessages((m) => [
                ...m,
                {
                  id,
                  kind: 'agent',
                  role: agent,
                  emoji: meta.emoji,
                  color: meta.color,
                  text: '',
                },
              ])
            },
            onAgentTextDelta: (agent, delta) => {
              const targetId = streamingMsgIdsRef.current[agent]
              if (!targetId) return
              setQaMessages((m) =>
                m.map((msg) => (msg.id === targetId ? { ...msg, text: `${msg.text}${delta}` } : msg)),
              )
            },
            onAgentMessageDone: (agent) => {
              delete streamingMsgIdsRef.current[agent]
            },
            onFinalPayload: (payload) => {
              const ran = payload.execute.query ?? payload.generated_sql
              setSqlLog(ran)
            },
            onStreamError: (message) => {
              setSqlLog(`// Error: ${message}`)
            },
          },
        )
        setSqlLog('// Generating SQL query…')
        const ran = result.execute.query ?? result.generated_sql
        setSqlLog(ran)
        setAgentStates(AGENTS.map(() => 'idle'))
        if (!streamedAgentMessages) {
          const insights =
            result.agent_messages && result.agent_messages.length > 0
              ? result.agent_messages
              : [{ role: 'ANALYST' as const, text: result.answer }]
          setQaMessages((m) => [...m, ...insights.map((ins) => insightToMessage(ins, result.generated_sql))])
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
        streamingMsgIdsRef.current = {}
        setSqlLog(`// Error: ${msg}`)
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
    sqlLog,
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
