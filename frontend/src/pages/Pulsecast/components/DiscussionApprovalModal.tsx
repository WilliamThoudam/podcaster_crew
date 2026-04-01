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
          ? 'Runs one round with all agents, then gives the final answer.'
          : 'Runs a multi-round panel discussion before the final answer.'
        : 'Continue the discussion to improve or validate the answer.';

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

