import { Children, isValidElement, type ReactElement, type ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import { PulsecastTable } from './PulsecastTable'
import { normalizePulsecastMarkdown } from '../utils'

type PulsecastMarkdownProps = {
  content: string
}

function isSqlHighlightedCode(children: ReactNode): boolean {
  const nodes = Children.toArray(children)
  if (nodes.length !== 1 || !isValidElement(nodes[0])) return false
  const el = nodes[0] as ReactElement<{ className?: string }>
  const cls = el.props?.className
  return typeof cls === 'string' && (cls.includes('language-sql') || /\bsql\b/i.test(cls))
}

export function PulsecastMarkdown({ content }: PulsecastMarkdownProps) {
  if (!content) return null
  const normalized = normalizePulsecastMarkdown(content)
  return (
    <div className="pulsecast-markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          table({ children }) {
            return <PulsecastTable>{children}</PulsecastTable>
          },
          thead({ children, ...props }) {
            return <thead {...props}>{children}</thead>
          },
          tbody({ children, ...props }) {
            return <tbody {...props}>{children}</tbody>
          },
          tr({ children, ...props }) {
            return <tr {...props}>{children}</tr>
          },
          th({ children, ...props }) {
            return <th {...props}>{children}</th>
          },
          td({ children, ...props }) {
            return <td {...props}>{children}</td>
          },
          pre({ children, className, ...props }) {
            if (isSqlHighlightedCode(children)) {
              return (
                <div className="sql-block markdown-sql-block">
                  <span className="sql-label">SQL QUERY GENERATED</span>
                  <pre
                    {...props}
                    className={['markdown-sql-pre-inner', className].filter(Boolean).join(' ')}
                  >
                    {children}
                  </pre>
                </div>
              )
            }
            return (
              <pre className={className} {...props}>
                {children}
              </pre>
            )
          },
        }}
      >
        {normalized}
      </ReactMarkdown>
    </div>
  )
}
