import { useLayoutEffect, useRef } from 'react'
import { Navigate, useNavigate, useParams } from 'react-router-dom'
import { paths } from '../../routes/paths'
import { type Screen, isPulsecastScreen } from '../../types'
import { AGENTS, PODCAST_SCRIPT, TOTAL_MS } from './constants'
import { formatQaClock } from './utils'
import { usePulsecastApp } from './hooks/usePulsecastApp'
import { PulsecastMarkdown } from './components/PulsecastMarkdown'
import { QaThinkingDots } from './components/QaThinkingDots'
import { DiscussionApprovalModal } from './components/DiscussionApprovalModal'
import './pulsecast.css'

const SCREENS: { id: Screen; icon: string; title: string }[] = [
  { id: 'dashboard', icon: '📊', title: 'Dashboard' },
  { id: 'player', icon: '🎙️', title: 'Podcast Player' },
  { id: 'transcript', icon: '📝', title: 'Transcript' },
  { id: 'interaction', icon: '💬', title: 'Ask Questions' },
]

export function Pulsecast() {
  const navigate = useNavigate()
  const { screen: screenParam } = useParams<{ screen: string }>()
  const validated: Screen | null = isPulsecastScreen(screenParam) ? screenParam : null
  const app = usePulsecastApp(validated ?? 'dashboard', navigate)
  const qaMessagesScrollRef = useRef<HTMLDivElement>(null)
  const qaMessagesInnerRef = useRef<HTMLDivElement>(null)
  /** When true, new content keeps the viewport pinned to the latest message (streaming). */
  const qaStickToBottomRef = useRef(true)

  const onQaMessagesScroll = () => {
    const el = qaMessagesScrollRef.current
    if (!el) return
    const slack = 96
    qaStickToBottomRef.current =
      el.scrollHeight - el.scrollTop - el.clientHeight <= slack
  }

  useLayoutEffect(() => {
    const el = qaMessagesScrollRef.current
    if (!el) return
    const last = app.qaMessages[app.qaMessages.length - 1]
    if (last?.kind === 'user') {
      qaStickToBottomRef.current = true
    }
    if (!qaStickToBottomRef.current) return
    el.scrollTop = el.scrollHeight
  }, [app.qaMessages, app.qaStreaming])

  useLayoutEffect(() => {
    const outer = qaMessagesScrollRef.current
    const inner = qaMessagesInnerRef.current
    if (!outer || !inner) return
    const ro = new ResizeObserver(() => {
      if (!qaStickToBottomRef.current) return
      outer.scrollTop = outer.scrollHeight
    })
    ro.observe(inner)
    return () => ro.disconnect()
  }, [app.qaMessages.length])

  if (validated === null) {
    return <Navigate to={paths.dashboard} replace />
  }

  return (
    <>
      <div className="app">
        <div className="sidebar">
          <div className="logo">PC</div>
          {SCREENS.map((s) => (
            <button
              key={s.id}
              type="button"
              className={`nav-btn${app.screen === s.id ? ' active' : ''}`}
              title={s.title}
              onClick={() => app.showScreen(s.id)}
            >
              {s.icon}
              {s.id === 'interaction' ? <span className="badge" /> : null}
            </button>
          ))}
        </div>

        <div className="main">
          <div className="topbar">
            <div className="live-dot" />
            <div className="topbar-title">{app.topbarTitle}</div>
            <div className="topbar-sub">{app.topbarSub}</div>
          </div>

          <div
            className={`screen${app.screen === 'dashboard' ? ' active' : ''}`}
            id="screen-dashboard"
          >
            <div className="section-label">KPI OVERVIEW</div>
            <div className="kpi-grid">
              <div className="kpi-card down">
                <div className="kpi-label">TOTAL SALES TODAY</div>
                <div className="kpi-value">$2.14M</div>
                <div className="kpi-change down">▼ 12.3% vs yesterday</div>
              </div>
              <div className="kpi-card up">
                <div className="kpi-label">NORTH REGION</div>
                <div className="kpi-value">$810K</div>
                <div className="kpi-change up">▲ 3.1%</div>
              </div>
              <div className="kpi-card alert">
                <div className="kpi-label">SOUTH REGION</div>
                <div className="kpi-value">$420K</div>
                <div className="kpi-change down">▼ 18.5% 🔴</div>
              </div>
              <div className="kpi-card neutral">
                <div className="kpi-label">ACTIVE PRODUCTS</div>
                <div className="kpi-value">47</div>
                <div className="kpi-change" style={{ color: 'var(--text3)' }}>
                  → no change
                </div>
              </div>
            </div>

            <div>
              <div className="section-label">REGION CONTRIBUTION</div>
              <div className="kpi-card" style={{ marginTop: 0 }}>
                <div className="region-bars">
                  {[
                    ['North', 78, 'var(--green)', '$810K'],
                    ['East', 56, 'var(--accent)', '$580K'],
                    ['West', 32, 'var(--accent3)', '$332K'],
                    ['South', 40, 'var(--red)', '$420K'],
                  ].map(([label, w, bg, val]) => (
                    <div className="region-row" key={String(label)}>
                      <div className="region-label">{label}</div>
                      <div className="region-bar-wrap">
                        <div
                          className="region-bar-fill"
                          style={{ width: `${w}%`, background: String(bg) }}
                        />
                      </div>
                      <div className="region-val">{val}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div>
              <div className="section-label">ACTIVE ALERTS</div>
              <div className="alert-list">
                <div
                  className="alert-card critical"
                  onClick={app.launchPodcast}
                  onKeyDown={(e) => e.key === 'Enter' && app.launchPodcast()}
                  role="button"
                  tabIndex={0}
                >
                  <div className="alert-icon red">🔴</div>
                  <div className="alert-body">
                    <div className="alert-title">South Region Sales Down 18.5%</div>
                    <div className="alert-desc">
                      Day-over-day comparison triggered · Product A driving 70% of decline
                    </div>
                  </div>
                  <div className="alert-time">2 min ago</div>
                  <button
                    type="button"
                    className="listen-btn"
                    onClick={(e) => {
                      e.stopPropagation()
                      app.launchPodcast()
                    }}
                  >
                    ▶ Listen
                  </button>
                </div>
                <div
                  className="alert-card warning"
                  onClick={() => app.showScreen('transcript')}
                  onKeyDown={(e) => e.key === 'Enter' && app.showScreen('transcript')}
                  role="button"
                  tabIndex={0}
                >
                  <div className="alert-icon gold">📉</div>
                  <div className="alert-body">
                    <div className="alert-title">WoW Trend Declining — 8% Drop</div>
                    <div className="alert-desc">
                      Week-over-week signal detected · Consistent 5-day decline
                    </div>
                  </div>
                  <div className="alert-time">14 min ago</div>
                  <button
                    type="button"
                    className="listen-btn"
                    style={{
                      background: 'linear-gradient(135deg,var(--gold),var(--accent2))',
                    }}
                    onClick={(e) => {
                      e.stopPropagation()
                      app.showScreen('transcript')
                    }}
                  >
                    📋 View
                  </button>
                </div>
                <div className="alert-card info">
                  <div className="alert-icon blue">📊</div>
                  <div className="alert-body">
                    <div className="alert-title">Product B Gaining Share</div>
                    <div className="alert-desc">
                      Contribution analysis shows +4.2% shift · Marketing campaign active
                    </div>
                  </div>
                  <div className="alert-time">1 hr ago</div>
                  <button
                    type="button"
                    className="listen-btn"
                    style={{
                      background: 'linear-gradient(135deg,var(--accent3),var(--accent))',
                    }}
                  >
                    ▶ Listen
                  </button>
                </div>
              </div>
            </div>

            <div className="sparkline-row">
              <div className="spark-card">
                <div className="spark-title">7-Day Sales Trend</div>
                <svg className="spark-svg" viewBox="0 0 300 60" preserveAspectRatio="none">
                  <defs>
                    <linearGradient id="sg1" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#00d4ff" stopOpacity="0.4" />
                      <stop offset="100%" stopColor="#00d4ff" stopOpacity="0" />
                    </linearGradient>
                  </defs>
                  <path
                    d="M0,15 L43,18 L86,12 L129,20 L172,28 L215,35 L258,42 L300,50"
                    fill="none"
                    stroke="#00d4ff"
                    strokeWidth="2"
                  />
                  <path
                    d="M0,15 L43,18 L86,12 L129,20 L172,28 L215,35 L258,42 L300,50 L300,60 L0,60Z"
                    fill="url(#sg1)"
                  />
                </svg>
              </div>
              <div className="spark-card">
                <div className="spark-title">Product A vs B</div>
                <svg className="spark-svg" viewBox="0 0 300 60" preserveAspectRatio="none">
                  <path
                    d="M0,10 L60,15 L120,22 L180,30 L240,38 L300,48"
                    fill="none"
                    stroke="#ff4d6d"
                    strokeWidth="2"
                  />
                  <path
                    d="M0,50 L60,45 L120,38 L180,32 L240,26 L300,20"
                    fill="none"
                    stroke="#00e5a0"
                    strokeWidth="2"
                  />
                </svg>
              </div>
            </div>
          </div>

          <div className={`screen${app.screen === 'player' ? ' active' : ''}`} id="screen-player">
            <div className="player-body">
              <div className="player-left">
                <div className="now-playing-header">
                  <div className="podcast-art">🎙️</div>
                  <div className="podcast-meta">
                    <div className="podcast-ep">EPISODE · LIVE · EP-2024-0325</div>
                    <div className="podcast-title">
                      South Region Sales Decline
                      <br />— Root Cause Analysis
                    </div>
                    <div className="podcast-agents" style={{ marginTop: 10 }}>
                      {[
                        ['Host', 'var(--host)', 'rgba(0,212,255,0.08)'],
                        ['Analyst', 'var(--analyst)', 'rgba(123,97,255,0.08)'],
                        ['Aggregation', 'var(--aggregation)', 'rgba(167,139,250,0.10)'],
                        ['Marketing', 'var(--marketing)', 'rgba(255,107,53,0.08)'],
                        ['Finance', 'var(--finance)', 'rgba(0,229,160,0.08)'],
                        ['Web Crawler', 'var(--web-crawler)', 'rgba(126,184,218,0.12)'],
                        ['Challenger', 'var(--challenger)', 'rgba(245,200,66,0.08)'],
                      ].map(([name, col, bg]) => (
                        <span
                          key={String(name)}
                          className="agent-chip"
                          style={{
                            color: String(col),
                            borderColor: String(col),
                            background: String(bg),
                          }}
                        >
                          {name}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="waveform-container">
                  <div className="waveform-label">AUDIO WAVEFORM</div>
                  <div
                    className="waveform"
                    onClick={(e) =>
                      app.onWaveformClick(e.clientX, e.currentTarget.getBoundingClientRect())
                    }
                    role="presentation"
                  >
                    {app.waveformHeights.map((h, i) => (
                      <div
                        key={i}
                        className={`waveform-bar${
                          i < app.playedBarCount
                            ? ' played'
                            : i === app.playedBarCount
                              ? ' active'
                              : ''
                        }`}
                        style={{ height: `${h}px` }}
                      />
                    ))}
                  </div>
                </div>

                <div className="speaker-display">
                  <div className="speaker-now">
                    <div
                      className="speaker-avatar"
                      style={{
                        fontSize: 22,
                        borderColor: app.speakerSeg?.color ?? 'var(--host)',
                        background: app.speakerSeg
                          ? `rgba(${app.colorToRgb(app.speakerSeg.color)},0.08)`
                          : 'rgba(0,212,255,0.08)',
                      }}
                    >
                      {app.speakerSeg?.emoji ?? '🎤'}
                    </div>
                    <div className="speaker-info">
                      <div
                        className="speaker-role"
                        style={{ color: app.speakerSeg?.color ?? 'var(--host)' }}
                      >
                        {app.showSpeakerPlaceholder ? 'HOST' : app.speakerSeg?.role}
                      </div>
                      <div className="speaker-line">
                        {app.showSpeakerPlaceholder
                          ? 'Press play to start the podcast…'
                          : app.speakerSeg?.text}
                      </div>
                    </div>
                    <div className="speaking-bars">
                      {[0, 1, 2, 3].map((i) => (
                        <div
                          key={i}
                          className={`speaking-bar${app.speakingActive ? '' : ' inactive'}`}
                        />
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <div className="player-right">
                <div className="panel-header">📋 Live Transcript</div>
                <div className="transcript-feed">
                  {app.feedItems.map((item, idx) => {
                    const seg = PODCAST_SCRIPT[item.segmentIndex]
                    const state = idx === app.feedItems.length - 1 ? 'active' : 'played'
                    return (
                      <div key={item.id} className={`t-entry ${state}`}>
                        <div className="t-dot" style={{ background: seg.color }} />
                        <div className="t-content">
                          <div className="t-speaker" style={{ color: seg.color }}>
                            {seg.role}
                          </div>
                          <div className="t-text">{seg.text}</div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>

            <div className="player-controls">
              <button type="button" className="ctrl-btn" title="Skip back 10s" onClick={app.skipBack}>
                ⏮
              </button>
              <button type="button" className="ctrl-btn play-pause" onClick={app.togglePlay}>
                {app.isPlaying ? '⏸' : '▶'}
              </button>
              <button type="button" className="ctrl-btn" title="Skip forward 10s" onClick={app.skipFwd}>
                ⏭
              </button>
              <div className="time-display">
                {`${app.fmt(Math.floor(app.totalElapsed / 1000))} / ${app.fmt(Math.floor(TOTAL_MS / 1000))}`}
              </div>
              <div
                className="progress-bar"
                onClick={(e) =>
                  app.onProgressClick(e.clientX, e.currentTarget.getBoundingClientRect())
                }
                role="presentation"
              >
                <div className="progress-fill" style={{ width: `${app.progressPct}%` }} />
              </div>
              <button type="button" className="interrupt-btn" onClick={app.openInterrupt}>
                ⚡ Interrupt & Ask
              </button>
            </div>
          </div>

          <div
            className={`screen${app.screen === 'transcript' ? ' active' : ''}`}
            id="screen-transcript"
          >
            <div className="transcript-main">
              <div>
                <div
                  style={{
                    fontFamily: 'Syne,sans-serif',
                    fontSize: 22,
                    fontWeight: 700,
                    marginBottom: 4,
                  }}
                >
                  South Region Sales Decline
                </div>
                <div
                  style={{
                    fontSize: 12,
                    color: 'var(--text3)',
                    fontFamily: "'DM Mono',monospace",
                    marginBottom: 24,
                  }}
                >
                  Episode · {new Date().toLocaleDateString()} · 5 agents · 3:20 runtime
                </div>
              </div>
              {PODCAST_SCRIPT.map((seg, i) => {
                const showSql = seg.role === 'ANALYST' && i < 4
                const tSec = Math.floor(app.cumulativeMsBeforeSegment(i) / 1000)
                return (
                  <div
                    key={`${seg.role}-${i}`}
                    className="chat-bubble"
                    style={{ animationDelay: `${i * 0.05}s` }}
                  >
                    <div
                      className="bubble-avatar"
                      style={{
                        color: seg.color,
                        borderColor: seg.color,
                        background: `rgba(${app.colorToRgb(seg.color)},0.08)`,
                      }}
                    >
                      {seg.emoji}
                    </div>
                    <div className="bubble-body">
                      <div className="bubble-meta">
                        <span className="bubble-name" style={{ color: seg.color }}>
                          {seg.role}
                        </span>
                        <span className="bubble-time">{app.fmt(tSec)}</span>
                      </div>
                      <div className="bubble-text">
                        {seg.text}
                        {showSql ? (
                          <div className="sql-block">
                            <span className="sql-label">SQL QUERY GENERATED</span>
                            <span>{app.analystSqlSnippet}</span>
                          </div>
                        ) : null}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          <div
            className={`screen${app.screen === 'interaction' ? ' active' : ''}`}
            id="screen-interaction"
          >
            <div className="interaction-body">
              <div className="qa-panel">
                <div
                  className="qa-messages"
                  ref={qaMessagesScrollRef}
                  onScroll={onQaMessagesScroll}
                >
                  <div className="qa-messages-inner" ref={qaMessagesInnerRef}>
                    {app.qaMessages.map((m) => (
                      <div key={m.id} className="chat-bubble">
                        <div
                          className="bubble-avatar"
                          style={{
                            color: m.color,
                            borderColor: m.color,
                            background: `rgba(${app.colorToRgb(m.color)},0.08)`,
                          }}
                        >
                          {m.emoji}
                        </div>
                        <div className="bubble-body">
                          <div className="bubble-meta">
                            <span className="bubble-name" style={{ color: m.color }}>
                              {m.role}
                            </span>
                            {m.createdAt != null ? (
                              <span className="bubble-time">{formatQaClock(m.createdAt)}</span>
                            ) : null}
                          </div>
                          <div
                            className={`bubble-text${m.kind === 'user' ? ' user-bubble' : ''}`}
                          >
                            {m.kind === 'user' ? m.text : <PulsecastMarkdown content={m.text} />}
                          </div>
                        </div>
                      </div>
                    ))}
                    {app.qaStreaming ? (
                      <div className="qa-thinking-row">
                        <QaThinkingDots />
                      </div>
                    ) : null}
                  </div>
                </div>
                <div className="qa-input-bar">
                  <button
                    type="button"
                    className={`voice-btn${app.voiceRecording ? ' recording' : ''}`}
                    title="Voice input"
                    disabled={app.qaStreaming && !app.streamPaused}
                    onClick={app.toggleVoice}
                  >
                    {app.voiceRecording ? '⏹' : '🎤'}
                  </button>
                  {app.qaStreaming && app.streamJobId && !app.streamPaused ? (
                    <button
                      type="button"
                      className="qa-stream-btn"
                      title="Pause streaming (enables typing)"
                      onClick={() => void app.pulsecastSetStreamPaused(true)}
                    >
                      ⏸
                    </button>
                  ) : null}
                  {app.qaStreaming && app.streamJobId && app.streamPaused ? (
                    <button
                      type="button"
                      className="qa-stream-btn"
                      title="Resume streaming"
                      onClick={() => void app.pulsecastSetStreamPaused(false)}
                    >
                      ▶
                    </button>
                  ) : null}
                  <input
                    className="qa-input"
                    placeholder={
                      app.qaStreaming && !app.streamPaused
                        ? 'Agents are responding… pause to type'
                        : 'Ask about the data… e.g. Compare with last week'
                    }
                    value={app.qaInput}
                    disabled={app.qaStreaming && !app.streamPaused}
                    onChange={(e) => app.setQaInput(e.target.value)}
                    onKeyDown={(e) =>
                      e.key === 'Enter' &&
                      !(app.qaStreaming && !app.streamPaused) &&
                      app.sendQA()
                    }
                  />
                  <button
                    type="button"
                    className="qa-send"
                    disabled={app.qaStreaming && !app.streamPaused}
                    onClick={() => app.sendQA()}
                  >
                    ➤
                  </button>
                </div>
              </div>
              <div className="status-panel">
                <div>
                  <div className="status-section-title">Agent Status</div>
                  {AGENTS.map((agent, i) => {
                    const st = app.agentStates[i] ?? 'idle'
                    return (
                      <div key={agent.name} className="agent-status-row">
                        <div className={`agent-status-dot ${st}`} />
                        <div className="agent-status-name">
                          {agent.emoji} {agent.name}
                        </div>
                        <div className="agent-status-state">{st}</div>
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className={`interrupt-overlay${app.sqlHitlOpen ? ' show' : ''}`}>
        <div className="interrupt-modal sql-hitl-modal">
          <h3>
            {app.sqlHitlPauseKind === 'duplicate_sub_question'
              ? 'Approve revised sub-question'
              : app.sqlHitlPauseKind === 'web_search'
                ? 'Approve web search'
                : 'Approve follow-up SQL'}
          </h3>
          {app.sqlHitlPauseKind === 'web_search' &&
          app.sqlHitlWebSearchTotal != null &&
          app.sqlHitlWebSearchTotal > 1 ? (
            <p className="sql-hitl-rationale" style={{ color: 'var(--text2)', fontSize: '0.9rem' }}>
              Web search {app.sqlHitlWebSearchStep ?? '?'} of {app.sqlHitlWebSearchTotal}
            </p>
          ) : null}
          <p>
            {app.sqlHitlPauseKind === 'duplicate_sub_question'
              ? 'Text-to-SQL matched an earlier step. The analyst suggested a different sub-question. Approve to run it (you may edit), or decline to skip this step and continue the plan.'
              : app.sqlHitlPauseKind === 'web_search'
                ? 'Web Crawler proposed a public web search (via Serper) to add external context. Approve to run it (you may edit the query), or decline to continue with warehouse data only.'
                : 'Challenger proposed a follow-up analytic query for text-to-SQL. Approve to run it, or decline to finish with the data you already have.'}
          </p>
          {app.sqlHitlRationale ? (
            <p className="sql-hitl-rationale" style={{ color: 'var(--text2)', fontSize: '0.9rem' }}>
              <PulsecastMarkdown content={app.sqlHitlRationale} />
            </p>
          ) : null}
          <label className="section-label" style={{ marginTop: 8 }}>
            {app.sqlHitlPauseKind === 'web_search' ? 'Search query (editable)' : 'Sub-question (editable)'}
          </label>
          <textarea
            className="interrupt-input"
            style={{ minHeight: 100, resize: 'vertical' }}
            value={app.sqlHitlEdited}
            onChange={(e) => app.setSqlHitlEdited(e.target.value)}
            placeholder={app.sqlHitlProposed}
          />
          <div className="interrupt-actions">
            <button
              type="button"
              className="btn-primary"
              disabled={app.qaStreaming}
              onClick={() => app.submitSqlHitl(true)}
            >
              {app.sqlHitlPauseKind === 'web_search' ? 'Approve & search' : 'Approve & run SQL'}
            </button>
            <button
              type="button"
              className="btn-secondary"
              disabled={app.qaStreaming}
              onClick={() => app.submitSqlHitl(false)}
            >
              Decline
            </button>
          </div>
        </div>
      </div>

      <DiscussionApprovalModal
        open={app.discussionHitlOpen}
        disabled={app.qaStreaming}
        stage={app.discussionHitlStage}
        requestedDepth={app.discussionHitlRequestedDepth}
        roundIndex={app.discussionHitlRoundIndex}
        maxRounds={app.discussionHitlMaxRounds}
        focusForNextRound={app.discussionHitlFocusForNextRound}
        rationale={app.discussionHitlRationale}
        onApprove={() => app.submitDiscussionHitl(true)}
        onDecline={() => app.submitDiscussionHitl(false)}
      />

      <div className={`interrupt-overlay${app.interruptOpen ? ' show' : ''}`}>
        <div className="interrupt-modal">
          <h3>⚡ Interrupt Podcast</h3>
          <p>The podcast will pause for you only. Ask your question and it will resume from the live point.</p>
          <input
            className="interrupt-input"
            placeholder="e.g. Compare South region with last week"
            value={app.interruptDraft}
            onChange={(e) => app.setInterruptDraft(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && app.submitInterrupt()}
          />
          <div className="interrupt-actions">
            <button type="button" className="btn-primary" onClick={app.submitInterrupt}>
              Ask Question
            </button>
            <button type="button" className="btn-secondary" onClick={app.closeInterrupt}>
              Cancel
            </button>
          </div>
        </div>
      </div>

      <div className={`toast${app.toast.visible ? ' show' : ''}`}>{app.toast.message}</div>
    </>
  )
}
