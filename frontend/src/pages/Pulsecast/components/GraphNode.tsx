import { Handle, Position } from '@xyflow/react'
import type { NodeProps, Node } from '@xyflow/react'
import type { GraphNodeStatus } from '../hooks/useGraphVisualization'

type TopologyKind = 'start' | 'end' | 'node' | 'agent' | 'gate'

type GraphNodeData = {
  label: string
  nodeType: TopologyKind
  status?: GraphNodeStatus
  agentRole?: string
}

type GraphNodeRF = Node<GraphNodeData, 'graphNode' | 'graphTerminal' | 'graphAgent' | 'graphGate'>

const STATUS_CLASS: Record<GraphNodeStatus, string> = {
  idle: 'gf-node--idle',
  running: 'gf-node--running',
  complete: 'gf-node--complete',
}

function roleClass(role: string | undefined): string {
  if (!role) return ''
  const safe = role.replace(/[^a-z0-9-]/gi, '')
  return safe ? `gf-node--role-${safe}` : ''
}

export function GraphNodeComponent({ data, type }: NodeProps<GraphNodeRF>) {
  const status: GraphNodeStatus = data.status ?? 'idle'
  const isTerminal = type === 'graphTerminal'
  const isGate = type === 'graphGate'
  const isAgent = type === 'graphAgent'

  const shape = isGate ? 'gf-node--gate' : isAgent ? 'gf-node--agent' : ''
  const role = isAgent ? roleClass(data.agentRole) : ''

  return (
    <div
      className={`gf-node ${isTerminal ? 'gf-node--terminal' : ''} ${shape} ${role} ${STATUS_CLASS[status]}`}
    >
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
