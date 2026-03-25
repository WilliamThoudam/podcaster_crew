import { DURATIONS, PODCAST_SCRIPT } from './constants'

export function fmt(s: number): string {
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

const COLOR_MAP: Record<string, string> = {
  'var(--host)': '0,212,255',
  'var(--analyst)': '123,97,255',
  'var(--marketing)': '255,107,53',
  'var(--finance)': '0,229,160',
  'var(--challenger)': '245,200,66',
}

export function colorToRgb(cssVar: string): string {
  return COLOR_MAP[cssVar] ?? '0,212,255'
}

export function segmentIndexAtElapsed(elapsedMs: number): number {
  let acc = 0
  for (let i = 0; i < DURATIONS.length; i++) {
    if (elapsedMs < acc + DURATIONS[i]) return i
    acc += DURATIONS[i]
  }
  return Math.max(0, PODCAST_SCRIPT.length - 1)
}

export function cumulativeMsBeforeSegment(segmentIndex: number): number {
  return DURATIONS.slice(0, segmentIndex).reduce((a, b) => a + b, 0)
}
