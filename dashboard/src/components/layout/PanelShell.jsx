import { cn } from '@/lib/utils'

export default function PanelShell({ title, badge, actions, children, className }) {
  return (
    <div className={cn(
      'flex flex-col border overflow-hidden',
      'bg-[var(--dash-surface)] border-[var(--dash-border)]',
      className
    )}>
      <header className="flex items-center justify-between px-3 py-2 border-b border-[var(--dash-border)] bg-[var(--dash-bg)] shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-[var(--dash-text-head)] text-xs font-semibold uppercase tracking-widest font-mono">
            {title}
          </span>
          {badge && (
            <span className="text-[10px] px-1.5 py-0.5 bg-white/5 text-[var(--dash-text)] border border-[var(--dash-border-bright)] font-mono">
              {badge}
            </span>
          )}
        </div>
        {actions && <div className="flex items-center gap-2">{actions}</div>}
      </header>
      <div className="flex-1 overflow-hidden min-h-0 h-full">
        {children}
      </div>
    </div>
  )
}
