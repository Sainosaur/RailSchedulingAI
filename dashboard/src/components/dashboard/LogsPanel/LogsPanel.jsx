import { useEffect, useRef } from 'react'
import PanelShell from '@/components/layout/PanelShell'

const LEVEL_STYLE = {
  ERROR: 'text-[var(--dash-red)]',
  WARN:  'text-[var(--dash-orange)]',
  INFO:  'text-[var(--dash-text-bright)]',
  DEBUG: 'text-[var(--dash-text)]/50',
}

export default function LogsPanel({ logs }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  return (
    <PanelShell title="Logs" className="h-full">
      <div className="h-full overflow-y-auto dash-scroll px-3 py-2 flex flex-col gap-0.5">
        {logs.map((log) => (
          <div key={log.id} className="flex gap-2 text-[10px] font-mono leading-5 whitespace-nowrap">
            <span className="text-[var(--dash-text)]/50 shrink-0">[{log.timestamp}]</span>
            <span className={`w-10 shrink-0 font-semibold ${LEVEL_STYLE[log.level] ?? LEVEL_STYLE.INFO}`}>
              {log.level}
            </span>
            <span className="w-8 shrink-0 text-[var(--dash-text)]">{log.source}</span>
            <span className="text-[var(--dash-text-bright)] truncate">{log.message}</span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </PanelShell>
  )
}
