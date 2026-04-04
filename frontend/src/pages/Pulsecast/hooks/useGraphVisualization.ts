import { useCallback, useEffect, useRef, useState } from 'react'
import type { Node, Edge } from '@xyflow/react'
import dagre from '@dagrejs/dagre'
import type { StreamProgressEvent } from '../../../services/api/pulsecastQa'

export type GraphNodeStatus = 'idle' | 'running' | 'complete'

interface TopologyNode {
  id: string
  type: 'start' | 'end' | 'node' | 'agent' | 'gate'
}

interface TopologyEdge {
  source: string
  target: string
  conditional: boolean
  condition_label: string | null
}

interface TopologyResponse {
  nodes: TopologyNode[]
  edges: TopologyEdge[]
}

const NODE_WIDTH = 184
const NODE_HEIGHT = 64

function applyDagreLayout(
  nodes: Node[],
  edges: Edge[],
): { nodes: Node[]; edges: Edge[] } {
  const g = new dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: 'TB', ranksep: 96, nodesep: 56 })

  for (const node of nodes) {
    g.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT })
  }
  for (const edge of edges) {
    g.setEdge(edge.source, edge.target)
  }

  dagre.layout(g)

  const laidOut = nodes.map((node) => {
    const pos = g.node(node.id)
    return {
      ...node,
      position: {
        x: pos.x - NODE_WIDTH / 2,
        y: pos.y - NODE_HEIGHT / 2,
      },
    }
  })

  return { nodes: laidOut, edges }
}

function prettyLabel(id: string): string {
  if (id === '__start__') return 'Start'
  if (id === '__end__' || id.endsWith(':__end__')) return 'End'
  const inner = id.includes(':') ? id.slice(id.indexOf(':') + 1) : id
  return inner
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ')
}

/** CSS modifier for agent border colors (matches pulsecast.css role vars). */
function agentRoleModifier(id: string): string | undefined {
  if (!id.startsWith('agents:')) return undefined
  const s = id.slice('agents:'.length)
  if (s === 'host_finalize') return 'host'
  if (s === 'web_crawler') return 'web-crawler'
  if (s === 'init_agents') return 'init'
  return s.replace(/_/g, '-')
}

function topologyToRfType(t: TopologyNode['type']): string {
  if (t === 'start' || t === 'end') return 'graphTerminal'
  if (t === 'gate') return 'graphGate'
  if (t === 'agent') return 'graphAgent'
  return 'graphNode'
}

function buildReactFlowElements(topology: TopologyResponse) {
  const rfNodes: Node[] = topology.nodes.map((n) => ({
    id: n.id,
    type: topologyToRfType(n.type),
    data: {
      label: prettyLabel(n.id),
      nodeType: n.type,
      agentRole: n.type === 'agent' ? agentRoleModifier(n.id) : undefined,
    },
    position: { x: 0, y: 0 },
  }))

  const rfEdges: Edge[] = topology.edges.map((e, i) => ({
    id: `e-${i}`,
    source: e.source,
    target: e.target,
    label: e.condition_label ?? undefined,
    type: 'smoothstep',
    animated: false,
    style: e.conditional
      ? { strokeDasharray: '6 3', strokeWidth: 2 }
      : { strokeWidth: 2 },
  }))

  return applyDagreLayout(rfNodes, rfEdges)
}

function apiBase(): string {
  const b = import.meta.env.VITE_PULSECAST_API_URL
  return (typeof b === 'string' && b.length > 0 ? b : 'http://localhost:8000').replace(/\/$/, '')
}

export function useGraphVisualization() {
  const [nodes, setNodes] = useState<Node[]>([])
  const [edges, setEdges] = useState<Edge[]>([])
  const [nodeStatusMap, setNodeStatusMap] = useState<Record<string, GraphNodeStatus>>({})
  const [isLoading, setIsLoading] = useState(true)
  const topologyRef = useRef<TopologyResponse | null>(null)

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const res = await fetch(`${apiBase()}/v1/graph/topology`, { cache: 'no-store' })
        if (!res.ok) throw new Error(`topology fetch failed: ${res.status}`)
        const data: TopologyResponse = await res.json()
        if (cancelled) return
        topologyRef.current = data
        const { nodes: n, edges: e } = buildReactFlowElements(data)
        setNodes(n)
        setEdges(e)

        const initial: Record<string, GraphNodeStatus> = {}
        for (const nd of data.nodes) {
          initial[nd.id] = 'idle'
        }
        setNodeStatusMap(initial)
      } catch (err) {
        console.error('Failed to load graph topology', err)
      } finally {
        if (!cancelled) setIsLoading(false)
      }
    }
    load()
    return () => {
      cancelled = true
    }
  }, [])

  const resetStatus = useCallback(() => {
    if (!topologyRef.current) return
    const fresh: Record<string, GraphNodeStatus> = {}
    for (const nd of topologyRef.current.nodes) {
      fresh[nd.id] = 'idle'
    }
    setNodeStatusMap(fresh)
  }, [])

  const onGraphProgress = useCallback((event: StreamProgressEvent) => {
    if (event.type === 'graph_node_entered') {
      setNodeStatusMap((prev) => ({ ...prev, [event.node]: 'running' }))
    } else if (event.type === 'graph_node_exited') {
      setNodeStatusMap((prev) => ({ ...prev, [event.node]: 'complete' }))
    }
  }, [])

  return { nodes, edges, nodeStatusMap, onGraphProgress, resetStatus, isLoading }
}
