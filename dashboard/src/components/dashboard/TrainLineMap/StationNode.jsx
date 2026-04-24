import { Handle, Position } from '@xyflow/react'

export default function StationNode({ data }) {
  return (
    <div className="relative flex flex-col items-center" style={{ width: 48 }}>
      <Handle
        type="target"
        position={Position.Left}
        style={{ opacity: 0, width: 0, height: 0, minWidth: 0, minHeight: 0, border: 'none' }}
      />
      <Handle
        type="source"
        position={Position.Right}
        style={{ opacity: 0, width: 0, height: 0, minWidth: 0, minHeight: 0, border: 'none' }}
      />

      <div className="w-3 h-3 rounded-full bg-[var(--dash-text-bright)] border-2 border-[var(--dash-border-bright)]" />

      <span className="mt-1.5 text-[9px] font-mono text-[var(--dash-text-bright)] tracking-widest uppercase select-none whitespace-nowrap">
        {data.label}
      </span>
    </div>
  )
}
