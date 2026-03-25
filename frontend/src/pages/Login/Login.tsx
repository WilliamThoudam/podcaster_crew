import { Link, useNavigate } from 'react-router-dom'
import { paths } from '../../routes/paths'

export function Login() {
  const navigate = useNavigate()

  return (
    <div
      style={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 16,
        fontFamily: "'DM Sans', system-ui, sans-serif",
        background: '#080c14',
        color: '#e8f0fe',
      }}
    >
      <h1 style={{ fontFamily: 'Syne, sans-serif', fontSize: 28 }}>Pulsecast</h1>
      <p style={{ color: '#8ba4c0', maxWidth: 360, textAlign: 'center' }}>
        Sign-in is a placeholder. Wire this route to your auth provider when ready.
      </p>
      <button
        type="button"
        onClick={() => navigate(paths.dashboard, { replace: true })}
        style={{
          padding: '12px 28px',
          borderRadius: 10,
          border: 'none',
          fontWeight: 600,
          cursor: 'pointer',
          background: 'linear-gradient(135deg, #00d4ff, #7b61ff)',
          color: '#000',
        }}
      >
        Continue to app
      </button>
      <Link to={paths.dashboard} style={{ color: '#00d4ff', fontSize: 14 }}>
        Or go to dashboard →
      </Link>
    </div>
  )
}
