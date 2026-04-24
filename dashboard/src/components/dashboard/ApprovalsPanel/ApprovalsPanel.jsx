import { useState, useEffect } from 'react'
import PanelShell from '@/components/layout/PanelShell'

const SEVERITY_STYLE = {
  critical: "text-[var(--dash-red)] border-[var(--dash-red)]/40 bg-[var(--dash-red)]/10",
  warning: "text-[var(--dash-orange)] border-[var(--dash-orange)]/40 bg-[var(--dash-orange)]/10",
  info: "text-[var(--dash-text)] border-[var(--dash-border-bright)] bg-white/5",
}

export default function ApprovalsPanel({ recommendations, onApprove, onDeny }) {
  const pending = recommendations.filter(r => r.approval && r.status === 'pending')

  const badge = pending.length > 0 ? String(pending.length) : null

  return (
    <PanelShell title="Approval Queue" badge={badge}>
      <div className="h-full overflow-y-auto dash-scroll">
        {pending.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <span className="text-[var(--dash-text)]/40 text-xs font-mono uppercase tracking-widest">
              No pending approvals
            </span>
          </div>
        ) : (
          pending.map(rec => (
            <ApprovalItem key={rec.id} rec={rec} onApprove={onApprove} onDeny={onDeny} />
          ))
        )}
      </div>
    </PanelShell>
  )
}

function ApprovalItem({ rec, onApprove, onDeny }) {
  const [confirmState, setConfirmState] = useState(null) // null | 'approve' | 'deny'

  useEffect(() => {
    if (!confirmState) return
    const t = setTimeout(() => setConfirmState(null), 5000)
    return () => clearTimeout(t)
  }, [confirmState])

  function handleApproveClick() {
    if (confirmState === 'approve') {
      onApprove(rec.id)
    } else {
      setConfirmState('approve')
    }
  }

  function handleDenyClick() {
    if (confirmState === 'deny') {
      onDeny(rec.id)
    } else {
      setConfirmState('deny')
    }
  }

  return (
    <div className="flex flex-col gap-2 px-3 py-3 border-b border-[var(--dash-border)] hover:bg-white/[0.02] transition-colors">
      <div className="flex items-center gap-2 flex-wrap">
        <span className={`text-[9px] font-mono font-semibold px-1.5 py-0.5 border uppercase tracking-wider ${SEVERITY_STYLE[rec.severity]}`}>
          {rec.severity}
        </span>
        <span className="text-[9px] font-mono px-1.5 py-0.5 border border-[var(--dash-border-bright)] text-[var(--dash-text-bright)] uppercase tracking-wider">
          {rec.station}
        </span>
        <span className="text-[9px] font-mono px-1.5 py-0.5 border border-[var(--dash-border-bright)] text-[var(--dash-text)] uppercase tracking-wider">
          {rec.type}
        </span>
        <span className="ml-auto text-[9px] font-mono text-[var(--dash-text)]/60">
          {rec.id}
        </span>
      </div>

      <p className="text-[var(--dash-text-bright)] text-xs leading-relaxed">
        {rec.message}
      </p>

      {confirmState && (
        <p className="text-[9px] font-mono text-[var(--dash-amber)] uppercase tracking-widest">
          Click again to confirm {confirmState}
        </p>
      )}

      <div className="flex gap-2">
        <button
          onClick={handleApproveClick}
          className={`flex-1 py-1.5 text-[10px] font-mono uppercase tracking-widest border transition-all duration-150 ${
            confirmState === 'approve'
              ? 'border-[var(--dash-green)] bg-[var(--dash-green)]/20 text-[var(--dash-green)] animate-pulse'
              : 'border-[var(--dash-green)]/40 bg-[var(--dash-green)]/5 text-[var(--dash-green)] hover:bg-[var(--dash-green)]/15'
          }`}
        >
          {confirmState === 'approve' ? 'Confirm' : 'Approve'}
        </button>
        <button
          onClick={handleDenyClick}
          className={`flex-1 py-1.5 text-[10px] font-mono uppercase tracking-widest border transition-all duration-150 ${
            confirmState === 'deny'
              ? 'border-[var(--dash-red)] bg-[var(--dash-red)]/20 text-[var(--dash-red)] animate-pulse'
              : 'border-[var(--dash-red)]/40 bg-[var(--dash-red)]/5 text-[var(--dash-red)] hover:bg-[var(--dash-red)]/15'
          }`}
        >
          {confirmState === 'deny' ? 'Confirm' : 'Deny'}
        </button>
      </div>
    </div>
  )
}
