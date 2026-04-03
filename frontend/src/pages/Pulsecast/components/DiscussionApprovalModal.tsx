import { PulsecastMarkdown } from './PulsecastMarkdown'

type DiscussionApprovalModalProps = {
  open: boolean
  disabled?: boolean
  stage: 'pre' | 'mid'
  requestedDepth: 'linear' | 'moderated' | null
  roundIndex: number
  maxRounds: number
  focusForNextRound: string | null
  rationale: string | null
  onApprove: () => void
  onDecline: () => void
}

export function DiscussionApprovalModal(props: DiscussionApprovalModalProps) {
  const {
    open,
    disabled,
    stage,
    requestedDepth,
    roundIndex,
    maxRounds,
    focusForNextRound,
    rationale,
    onApprove,
    onDecline,
  } = props

  if (!open) return null

  const title =
    stage === 'pre'
      ? requestedDepth === 'linear'
        ? 'Start Discussion'
        : 'Start Panel Discussion'
      : `Continue to Round ${roundIndex} of ${maxRounds}`

  const body =
    stage === 'pre'
      ? requestedDepth === 'linear'
        ? 'This will run a discussion (Analyst, Marketing, Finance, Forecaster, Web Crawler, Challenger) before the Host final answer.'
        : 'This will run a panel discussion (Analyst, Marketing, Finance, Forecaster, Web Crawler, Challenger) that may continue for multiple rounds before the Host final answer.'
      : 'Challenger recommends continuing the discussion to deepen or validate the analysis.'

  return (
    <div className={`interrupt-overlay${open ? ' show' : ''}`}>
      <div className="interrupt-modal">
        <h3>{title}</h3>
        <p>{body}</p>
        {focusForNextRound ? (
          <p className="sql-hitl-rationale" style={{ color: 'var(--text2)', fontSize: '0.9rem' }}>
            <strong>Focus:</strong> {focusForNextRound}
          </p>
        ) : null}
        {rationale ? (
          <p className="sql-hitl-rationale" style={{ color: 'var(--text2)', fontSize: '0.9rem' }}>
            <PulsecastMarkdown content={rationale} />
          </p>
        ) : null}
        <div className="interrupt-actions">
          <button type="button" className="btn-primary" disabled={disabled} onClick={onApprove}>
            Approve
          </button>
          <button type="button" className="btn-secondary" disabled={disabled} onClick={onDecline}>
            Decline
          </button>
        </div>
      </div>
    </div>
  )
}

