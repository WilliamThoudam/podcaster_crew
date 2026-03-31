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
        ? 'Start 1-round panel'
        : 'Start multi-round panel'
      : `Continue to round ${roundIndex} of ${maxRounds}`

  const body =
    stage === 'pre'
      ? requestedDepth === 'linear'
        ? 'This will run a one-pass panel (Analyst, Marketing, Finance, Web Crawler, Challenger) before the Host final answer.'
        : 'This will run a multi-round moderated panel discussion before the Host final answer.'
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
            {rationale}
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

