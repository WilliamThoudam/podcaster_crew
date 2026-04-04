import type { Screen } from '../types'

/** URL segments for Pulsecast shell (no leading slash in values — use with navigate). */
export const paths = {
  login: '/login',
  dashboard: '/dashboard',
  player: '/player',
  transcript: '/transcript',
  interaction: '/interaction',
  graph: '/graph',
} as const satisfies Record<'login' | Screen, string>

export function pathForScreen(screen: Screen): string {
  return `/${screen}`
}
