import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import { PulsecastTable } from './PulsecastTable'

type PulsecastMarkdownProps = {
  content: string
}

export function PulsecastMarkdown({ content }: PulsecastMarkdownProps) {
  if (!content) return null
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
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}

