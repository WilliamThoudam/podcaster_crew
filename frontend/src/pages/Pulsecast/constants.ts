import type { PodcastLine } from '../../types'
import type { Screen } from '../../types'

export const PODCAST_SCRIPT: PodcastLine[] = [
  {
    role: 'HOST',
    emoji: '🎤',
    color: 'var(--host)',
    text: "We've detected a significant decline in sales in the South region. Sales are down 18.5% compared to yesterday — well above our alert threshold of 10%. Let's understand what's happening.",
  },
  {
    role: 'ANALYST',
    emoji: '📊',
    color: 'var(--analyst)',
    text: 'Sales dropped by 18.5% compared to yesterday, primarily driven by Product A. The South region went from $515K yesterday to $420K today — a drop of $95K.',
  },
  {
    role: 'HOST',
    emoji: '🎤',
    color: 'var(--host)',
    text: 'Can you break that down further? Which products are contributing most to this decline?',
  },
  {
    role: 'ANALYST',
    emoji: '📊',
    color: 'var(--analyst)',
    text: 'Product A is responsible for approximately 70% of the decline — about $66K of the drop. Product B remained stable, and Product C showed a slight 2% uptick. Compared to last week, sales are also down by 8%, indicating a sustained downward trend.',
  },
  {
    role: 'MARKETING',
    emoji: '📣',
    color: 'var(--marketing)',
    text: 'This aligns with a pause in our South region marketing campaigns. The campaign for Product A was placed on hold 3 days ago, which correlates directly with the start of the decline.',
  },
  {
    role: 'HOST',
    emoji: '🎤',
    color: 'var(--host)',
    text: 'Finance, what does this mean for our margins?',
  },
  {
    role: 'FINANCE',
    emoji: '💰',
    color: 'var(--finance)',
    text: 'Margins remain stable at 34% despite the volume drop. However, if the trend continues for 5 more days, we estimate a monthly revenue shortfall of approximately $450K to $500K.',
  },
  {
    role: 'CHALLENGER',
    emoji: '⚖️',
    color: 'var(--challenger)',
    text: 'Worth noting — this change is significantly higher than our normal daily variation, which averages 3 to 4 percent. This is not statistical noise. The 8% week-over-week signal confirms this is a real trend, not an anomaly.',
  },
  {
    role: 'HOST',
    emoji: '🎤',
    color: 'var(--host)',
    text: 'In summary — this appears to be a campaign-driven decline concentrated in Product A in the South region. Marketing should consider resuming the campaign, and this trend should be monitored closely over the next 48 hours.',
  },
]

export const DURATIONS = [5000, 6000, 4500, 7000, 5500, 4000, 6500, 5000, 5500] as const

export const TOTAL_MS = DURATIONS.reduce((a, b) => a + b, 0)

export const AGENTS = [
  { name: 'Host', emoji: '🎤', color: 'var(--host)' },
  { name: 'Analyst', emoji: '📊', color: 'var(--analyst)' },
  { name: 'Aggregation', emoji: '🧰', color: 'var(--aggregation)' },
  { name: 'Marketing', emoji: '📣', color: 'var(--marketing)' },
  { name: 'Finance', emoji: '💰', color: 'var(--finance)' },
  { name: 'Forecaster', emoji: '📈', color: 'var(--forecaster)' },
  { name: 'Web Crawler', emoji: '🕸️', color: 'var(--web-crawler)' },
  { name: 'Challenger', emoji: '⚖️', color: 'var(--challenger)' },
] as const

export const TOPBAR_BY_SCREEN: Record<Screen, [string, string]> = {
  dashboard: ['Dashboard — Live Monitoring', 'Monitoring 4 regions · 3 alerts'],
  player: ['Podcast Player — Live Episode', 'South Region Decline Analysis'],
  transcript: ['Full Transcript', 'Multi-agent conversation log'],
  interaction: ['Interaction Panel', 'Ask questions about your data'],
  graph: ['Graph Flow', 'Real-time LangGraph node status'],
}

export const ANALYST_SQL_SNIPPET = `SELECT region, SUM(sales_amount),
  ((today-yesterday)/yesterday*100.0) AS pct_change
FROM sales_fact
WHERE date IN (CURRENT_DATE, CURRENT_DATE-1)
GROUP BY region
HAVING ABS(pct_change) > 10;`
