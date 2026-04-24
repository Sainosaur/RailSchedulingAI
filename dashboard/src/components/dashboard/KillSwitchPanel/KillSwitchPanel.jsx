import { useState, useEffect } from "react";
import PanelShell from "@/components/layout/PanelShell";

export default function KillSwitchPanel({
  killed,
  reason,
  onKill,
  onRestore,
  onReasonChange,
}) {
  const [confirmPending, setConfirmPending] = useState(false);
  const [showReasonModal, setShowReasonModal] = useState(false);

  // Auto-reset confirm state after 5s
  useEffect(() => {
    if (!confirmPending) return;
    const t = setTimeout(() => setConfirmPending(false), 5000);
    return () => clearTimeout(t);
  }, [confirmPending]);

  function handleKillClick() {
    if (!confirmPending) {
      setConfirmPending(true);
    } else {
      setConfirmPending(false);
      onKill();
      setShowReasonModal(true);
    }
  }

  const statusBadge = (
    <span
      className={`text-[10px] font-mono px-1.5 py-0.5 border ${killed
        ? "text-[var(--dash-kill)] border-[var(--dash-kill)]/40 bg-[var(--dash-kill)]/10"
        : "text-[var(--dash-amber)] border-[var(--dash-amber)]/40 bg-[var(--dash-amber)]/10"
        }`}
    >
      {killed ? "KILLED" : "READY"}
    </span>
  );

  return (
    <>
      <PanelShell
        title="Kill Switch"
        badge=""
        actions={statusBadge}
        className="h-full"
      >
        <div className="flex flex-col gap-3 p-3 h-full overflow-y-auto dash-scroll">
          {/* Kill / Confirm button */}
          {!killed && (
            <button
              onClick={handleKillClick}
              className={`w-full py-4 px-4 border-2 font-bold tracking-[0.15em] uppercase text-sm font-mono transition-all duration-150 active:scale-[0.98] ${confirmPending
                ? "border-[var(--dash-kill)] bg-[var(--dash-kill)]/30 text-white shadow-[0_0_24px_var(--dash-kill-glow)] animate-pulse"
                : "border-[var(--dash-kill)] bg-[var(--dash-kill)]/10 text-[var(--dash-kill)] hover:bg-[var(--dash-kill)]/20 hover:shadow-[0_0_20px_var(--dash-kill-glow)]"
                }`}
            >
              {confirmPending ? "CONFIRM KILL" : "KILL SWITCH"}
            </button>
          )}

          {/* Killed state */}
          {killed && (
            <>
              <div className="border-2 border-[var(--dash-kill)] bg-[var(--dash-kill)]/10 p-4 shadow-[0_0_30px_var(--dash-kill-glow)]">
                <div className="flex items-center gap-2 mb-1">
                  <span className="w-2 h-2 rounded-full bg-[var(--dash-kill)] animate-pulse" />
                  <span className="text-[var(--dash-kill)] text-sm font-bold font-mono tracking-[0.1em] uppercase">
                    System Killed!
                  </span>
                </div>
                <p className="text-[var(--dash-text)] text-xs font-mono leading-relaxed">
                  All services halted.
                </p>
              </div>

              <button
                onClick={onRestore}
                className="w-full py-2 px-4 border border-[var(--dash-border-bright)] bg-transparent text-[var(--dash-text-bright)] text-xs font-mono uppercase tracking-widest hover:border-[var(--dash-green)] hover:text-[var(--dash-green)] transition-colors"
              >
                Restore Service
              </button>

              <button
                onClick={() => setShowReasonModal(true)}
                className="w-full py-2 px-4 border border-[var(--dash-border-bright)] bg-transparent text-[var(--dash-text)]/60 text-xs font-mono uppercase tracking-widest hover:border-[var(--dash-border-bright)] hover:text-[var(--dash-text-bright)] transition-colors"
              >
                Incident Report
              </button>
            </>
          )}
        </div>
      </PanelShell>

      {/* Incident report modal */}
      {showReasonModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="w-[480px] flex flex-col border-2 border-[var(--dash-kill)] bg-[var(--dash-surface)] shadow-[0_0_60px_var(--dash-kill-glow)]">
            <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--dash-border)] bg-[var(--dash-bg)]">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-[var(--dash-kill)] animate-pulse" />
                <span className="text-[var(--dash-kill)] text-xs font-bold font-mono tracking-[0.15em] uppercase">
                  Incident Report
                </span>
              </div>
              <span className="text-[10px] font-mono text-[var(--dash-text)]/40 uppercase tracking-widest">
                Required
              </span>
            </div>

            <div className="flex flex-col gap-3 p-4">
              <label className="text-[10px] font-mono text-[var(--dash-text)] uppercase tracking-widest">
                What went wrong:
              </label>
              <textarea
                value={reason}
                onChange={(e) => onReasonChange(e.target.value)}
                rows={6}
                placeholder="Describe the incident…"
                autoFocus
                className="w-full p-3 bg-[var(--dash-bg)] border border-[var(--dash-border-bright)] text-[var(--dash-text-bright)] text-xs font-mono leading-relaxed resize-none focus:outline-none focus:border-[var(--dash-kill)]/50 placeholder:text-[var(--dash-text)]/30"
              />
              <button
                onClick={() => setShowReasonModal(false)}
                className="w-full py-2.5 border border-[var(--dash-kill)]/40 bg-[var(--dash-kill)]/10 text-[var(--dash-kill)] text-xs font-mono uppercase tracking-widest hover:bg-[var(--dash-kill)]/20 transition-colors"
              >
                Submit Report
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
