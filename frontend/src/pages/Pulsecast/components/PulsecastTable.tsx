import type { ReactNode } from 'react'

type PulsecastTableProps = {
  children: ReactNode
}

export function PulsecastTable({ children }: PulsecastTableProps) {
  return (
    <div className="pulsecast-table-wrapper">
      <table className="pulsecast-table">{children}</table>
    </div>
  )
}

