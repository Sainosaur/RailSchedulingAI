import PanelShell from "@/components/layout/PanelShell";
import { RotateCcw } from "lucide-react";

const SEVERITY_STYLE = {
  ERROR:
    "text-[var(--dash-red)] border-[var(--dash-red)]/40 bg-[var(--dash-red)]/10",
  WARN: "text-[var(--dash-orange)] border-[var(--dash-orange)]/40 bg-[var(--dash-orange)]/10",
  INFO: "text-[var(--dash-text)] border-[var(--dash-border-bright)] bg-white/5",
  SAFETY:
    "text-[var(--dash-green)] border-[var(--dash-green)]/40 bg-[var(--dash-green)]/10",
};

export default function OverrideLogPanel({ logs, onReset }) {
  const actions = (
    <button
      onClick={onReset}
      className="p-1 hover:bg-white/10 rounded transition-colors group"
      title="Reset Logs"
    >
      <RotateCcw className="w-3.5 h-3.5 text-[var(--dash-text)] group-hover:text-[var(--dash-text-bright)]" />
    </button>
  );

  return (
    <PanelShell title="Override Log" actions={actions} className="h-full">
      <div className="h-full overflow-y-auto dash-scroll">
        {logs.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <span className="text-[var(--dash-text)]/40 text-[10px] font-mono uppercase tracking-widest">
              Log Empty
            </span>
          </div>
        ) : (
          [...logs].reverse().map((log) => (
            <div
              key={log.id}
              className="flex flex-col gap-1 px-3 py-2 border-b border-[var(--dash-border)] hover:bg-white/[0.02] transition-colors"
            >
              <div className="flex items-center gap-2">
                <span
                  className={`text-[9px] font-mono font-semibold px-1 py-0.5 border uppercase tracking-wider ${SEVERITY_STYLE[log.level] || SEVERITY_STYLE.INFO}`}
                >
                  {log.level}
                </span>
                <span className="text-[9px] font-mono text-[var(--dash-text)]/60">
                  {log.timestamp}
                </span>
                <span className="ml-auto text-[9px] font-mono text-[var(--dash-text)]/40 uppercase">
                  {log.source}
                </span>
              </div>
              <p className="text-[var(--dash-text-bright)] text-xs leading-tight font-mono">
                {log.message}
              </p>
            </div>
          ))
        )}
      </div>
    </PanelShell>
  );
}
