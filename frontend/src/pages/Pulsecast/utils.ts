import { DURATIONS, PODCAST_SCRIPT } from './constants'

export function fmt(s: number): string {
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

/** Short wall time for QA transcript headers (e.g. 3:05:42 PM). */
export function formatQaClock(ms: number): string {
  return new Date(ms).toLocaleTimeString(undefined, {
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
  })
}

/** True when content clearly crosses block boundaries (CommonMark emphasis cannot span these). */
function hasBlockMarkdownStructure(s: string): boolean {
  if (/\n\s*\n/.test(s)) return true
  if (/\n\s*(?:[-*+]|\d{1,3}\.)\s/.test(s)) return true
  return false
}

function tryStripOuterDelimiter(markdown: string, open: string, close: string): string | null {
  const trimmed = markdown.trim()
  if (trimmed.length < open.length + close.length + 1) return null
  if (!trimmed.startsWith(open) || !trimmed.endsWith(close)) return null
  if (open === '*' && /^\*\s/.test(trimmed)) return null
  const inner = trimmed.slice(open.length, trimmed.length - close.length)
  if (!hasBlockMarkdownStructure(inner)) return null
  const leading = markdown.match(/^\s*/)?.[0] ?? ''
  const trailing = markdown.match(/\s*$/)?.[0] ?? ''
  return leading + inner.replace(/^\s+/, '') + trailing
}

/**
 * Some model outputs try to wrap multi-line lists in `_..._`, but CommonMark emphasis cannot cross
 * block boundaries. That produces stray underscores like `_- item` or `..._` at line ends.
 * This pass removes *line-wrapper* underscores only when they appear as standalone wrappers
 * around list-like lines (keeps normal underscores inside words).
 */
function stripDanglingLineEmphasis(markdown: string): string {
  const lines = markdown.split('\n')
  const out = lines.map((line) => {
    // Preserve original indentation.
    const m = line.match(/^(\s*)(.*)$/)
    const indent = m?.[1] ?? ''
    let body = m?.[2] ?? line

    // Remove a leading "_" that directly precedes a list marker or em-dash bullet.
    // Examples: "_- A. ..."  "_* item"  "_— A. ..."  "_1. item"
    if (
      body.startsWith('_') &&
      /^_(?:\s*)(?:[-*+]|—|\d{1,3}\.)\s+/.test(body)
    ) {
      body = body.replace(/^_/, '')
    }

    // Remove a trailing "_" that appears to close a line-wrapper emphasis.
    // Example: "... averages._" or "..._)_" (we keep punctuation).
    if (/_\s*$/.test(body) && /[).,!?:;]_\s*$/.test(body)) {
      body = body.replace(/_\s*$/, '')
    }

    return indent + body
  })
  return out.join('\n')
}

/**
 * LLMs often wrap a whole section in `_..._` or `*...*` for emphasis, but CommonMark only allows
 * emphasis inside a single block — lists and paragraph breaks break pairing, so delimiters show as
 * literal characters. Strip one outer wrapper when it clearly spans blocks.
 */
export function normalizePulsecastMarkdown(markdown: string): string {
  if (!markdown) return markdown
  let s = markdown
  const pairs: [string, string][] = [
    ['__', '__'],
    ['**', '**'],
    ['_', '_'],
    ['*', '*'],
  ]
  for (let i = 0; i < 4; i++) {
    let changed = false
    for (const [open, close] of pairs) {
      const next = tryStripOuterDelimiter(s, open, close)
      if (next != null) {
        s = next
        changed = true
        break
      }
    }
    if (!changed) break
  }
  if (hasBlockMarkdownStructure(s)) {
    s = stripDanglingLineEmphasis(s)
  }
  return s
}

const COLOR_MAP: Record<string, string> = {
  'var(--host)': '0,212,255',
  'var(--analyst)': '123,97,255',
  'var(--marketing)': '255,107,53',
  'var(--finance)': '0,229,160',
  'var(--forecaster)': '56,189,248',
  'var(--web-crawler)': '126,184,218',
  'var(--challenger)': '245,200,66',
  'var(--accent)': '0,212,255',
  'var(--red)': '255,77,109',
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
