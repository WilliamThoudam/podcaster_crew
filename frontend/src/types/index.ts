export type Screen = 'dashboard' | 'player' | 'transcript' | 'interaction' | 'graph'

export const PULSECAST_SCREENS = [
  'dashboard',
  'player',
  'transcript',
  'interaction',
  'graph',
] as const satisfies readonly Screen[]

export function isPulsecastScreen(param: string | undefined): param is Screen {
  return param != null && (PULSECAST_SCREENS as readonly string[]).includes(param)
}

export type AgentState = 'idle' | 'thinking' | 'active'

export type PodcastRole =
  | 'HOST'
  | 'ANALYST'
  | 'AGGREGATION'
  | 'MARKETING'
  | 'FINANCE'
  | 'FORECASTER'
  | 'WEB_CRAWLER'
  | 'CHALLENGER'

export type PodcastLine = {
  role: PodcastRole
  emoji: string
  color: string
  text: string
}

export type QaMessage = {
  id: string
  kind: 'system' | 'user' | 'agent'
  role: string
  emoji: string
  color: string
  text: string
  sql?: string
  /** Wall-clock time when the row was created (transcript-style header). */
  createdAt?: number
}
