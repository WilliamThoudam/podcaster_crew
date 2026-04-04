import { Handle, Position } from '@xyflow/react'
import type { NodeProps, Node } from '@xyflow/react'
import type { GraphNodeStatus } from '../hooks/useGraphVisualization'

type GraphNodeData = {
  label: string
  nodeType: 'start' | 'end' | 'node'
  status?: GraphNodeStatus
}

type GraphNodeType = Node<GraphNodeData>

const STATUS_CLASS: Record<GraphNodeStatus, string> = {
  idle: 'gf-node--idle',
  running: 'gf-node--running',
  complete: 'gf-node--complete',
}

export function GraphNodeComponent({ data }: NodeProps<GraphNodeType>) {
  const status: GraphNodeStatus = data.status ?? 'idle'
  const isTerminal = data.nodeType === 'start' || data.nodeType === 'end'

  return (
    <div className={`gf-node ${isTerminal ? 'gf-node--terminal' : ''} ${STATUS_CLASS[status]}`}>
      {data.nodeType !== 'start' && (
        <Handle type="target" position={Position.Top} className="gf-handle" />
      )}
      <div className="gf-node__label">{data.label}</div>
      {data.nodeType !== 'end' && (
        <Handle type="source" position={Position.Bottom} className="gf-handle" />
      )}
    </div>
  )
}
