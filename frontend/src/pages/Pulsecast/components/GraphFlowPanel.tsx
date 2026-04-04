import { useMemo } from 'react'
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  ReactFlowProvider,
} from '@xyflow/react'
import type { Node, Edge } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { GraphNodeComponent } from './GraphNode'
import type { GraphNodeStatus } from '../hooks/useGraphVisualization'

const nodeTypes = {
  graphNode: GraphNodeComponent,
  graphTerminal: GraphNodeComponent,
} as const

interface GraphFlowPanelProps {
  nodes: Node[]
  edges: Edge[]
  nodeStatusMap: Record<string, GraphNodeStatus>
}

export function GraphFlowPanel({ nodes, edges, nodeStatusMap }: GraphFlowPanelProps) {
  const enrichedNodes = useMemo<Node[]>(
    () =>
      nodes.map((n) => ({
        ...n,
        data: { ...n.data, status: nodeStatusMap[n.id] ?? 'idle' },
      })),
    [nodes, nodeStatusMap],
  )

  const enrichedEdges = useMemo<Edge[]>(() => {
    const runningNodes = new Set(
      Object.entries(nodeStatusMap)
        .filter(([, s]) => s === 'running')
        .map(([id]) => id),
    )
    return edges.map((e) => ({
      ...e,
      animated: runningNodes.has(e.target),
    }))
  }, [edges, nodeStatusMap])

  return (
    <ReactFlowProvider>
      <div className="gf-panel">
        <ReactFlow
          nodes={enrichedNodes}
          edges={enrichedEdges}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.3 }}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={16} size={1} />
          <Controls showInteractive={false} />
          <MiniMap pannable zoomable />
        </ReactFlow>
      </div>
    </ReactFlowProvider>
  )
}
