import { useCallback, useEffect, useRef, useState } from 'react'
import type { Node, Edge } from '@xyflow/react'
import dagre from '@dagrejs/dagre'
import type { StreamProgressEvent } from '../../../services/api/pulsecastQa'

export type GraphNodeStatus = 'idle' | 'running' | 'complete'

interface TopologyNode {
  id: string
  type: 'start' | 'end' | 'node'
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

const NODE_WIDTH = 170
const NODE_HEIGHT = 60

function applyDagreLayout(
  nodes: Node[],
  edges: Edge[],
): { nodes: Node[]; edges: Edge[] } {
  const g = new dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: 'TB', ranksep: 80, nodesep: 50 })

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
  if (id === '__end__') return 'End'
  return id
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ')
}

function buildReactFlowElements(topology: TopologyResponse) {
  const rfNodes: Node[] = topology.nodes.map((n) => ({
    id: n.id,
    type: n.type === 'start' || n.type === 'end' ? 'graphTerminal' : 'graphNode',
    data: { label: prettyLabel(n.id), nodeType: n.type },
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
        const res = await fetch(`${apiBase()}/v1/graph/topology`)
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
