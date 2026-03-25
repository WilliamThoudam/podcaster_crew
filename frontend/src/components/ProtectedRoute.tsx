import type { ReactNode } from 'react'
import { Navigate, useLocation } from 'react-router-dom'
import { paths } from '../routes/paths'
import { isLoggedIn } from '../utils/auth'

type ProtectedRouteProps = { children: ReactNode }

export function ProtectedRoute({ children }: ProtectedRouteProps) {
  const location = useLocation()

  if (!isLoggedIn()) {
    return <Navigate to={paths.login} state={{ from: location }} replace />
  }

  return <>{children}</>
}
