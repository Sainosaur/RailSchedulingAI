import { useState, useEffect } from "react";
import PanelShell from "@/components/layout/PanelShell";
import Tooltip from "@/components/ui/Tooltip";
import { startSimulation, stopSimulation, simStatus } from "@/services/simulation";
import { stallLead, releaseLead, holdLead } from "@/services/LeadControl";

export default function SimulationControlPanel({ stalled, held }) {
  const [running, setRunning] = useState(false);

  // Initial state fetch
  useEffect(() => {
    simStatus()
      .then((data) => {
        setRunning(data.status);
      }).catch(() => {
        console.error("Failed to fetch lead status");
        setRunning(false);
      });
  }, []);


  function handleToggle() {
    if (running) {
      stopSimulation().then(() => {
        setRunning(false);
      }).catch(() => {
        console.error("Failed to stop simulation");
        setRunning(true);
      });
    } else {
      startSimulation().then(() => {
        setRunning(true);
      }).catch(() => {
        console.error("Failed to start simulation");
        setRunning(false);
      })
    }
  }

  const statusBadge = (
    <span
      className={`text-[10px] font-mono px-1.5 py-0.5 border ${running
        ? "text-[var(--dash-green)] border-[var(--dash-green)]/40 bg-[var(--dash-green)]/10"
        : "text-[var(--dash-text)] border-[var(--dash-border-bright)] bg-white/5"
        }`}
    >
      {running ? "RUNNING" : "STOPPED"}
    </span>
  );

  return (
    <PanelShell title="Simulation" badge="LINE 104" actions={statusBadge} className="h-full">
      <div className="flex flex-col gap-3 p-3 h-full dash-scroll overflow-y-auto">
        {/* Pulse indicator */}
        <div className="flex items-center gap-2 px-3 py-2 border border-[var(--dash-border)] bg-[var(--dash-bg)]">
          <span
            className={`w-2 h-2 rounded-full shrink-0 ${running
              ? "bg-[var(--dash-green)] animate-pulse shadow-[0_0_6px_var(--dash-green)]"
              : "bg-[var(--dash-border-bright)]"
              }`}
          />
          <span className="text-[10px] font-mono text-[var(--dash-text)] uppercase tracking-widest">
            {running ? "Simulation active" : "Awaiting start"}
          </span>
        </div>

        {/* Lead train status — only visible while running */}
        {running && (
          <div className="flex items-center gap-2 px-3 py-2 border border-[var(--dash-border)] bg-[var(--dash-bg)]">
            <span
              className={`w-2 h-2 rounded-full shrink-0 animate-pulse ${stalled
                ? "bg-[var(--dash-kill)] shadow-[0_0_6px_var(--dash-kill)]"
                : held
                  ? "bg-[var(--dash-amber)] shadow-[0_0_6px_var(--dash-amber)]"
                  : "bg-[var(--dash-green)] shadow-[0_0_6px_var(--dash-green)]"
                }`}
            />
            <span className={`text-[10px] font-mono uppercase tracking-widest ${stalled
              ? "text-[var(--dash-kill)]"
              : held
                ? "text-[var(--dash-amber)]"
                : "text-[var(--dash-text)]"
              }`}>
              {stalled ? "Emergency stalled" : held ? "Held / To be Held at station" : "Lead running"}
            </span>
          </div>
        )}

        {/* Lead train controls — only visible while running */}
        {running && (
          <div className="flex flex-col gap-2">
            <span className="text-[10px] font-mono text-[var(--dash-text)]/50 uppercase tracking-widest px-1">
              Lead Controls
            </span>
            <div className="grid grid-cols-2 gap-2">
              <Tooltip text="Emergency Stop Lead Train">
                <button
                  id="sim-stall-btn"
                  onClick={() => { stallLead() }}
                  className="w-full py-2 px-3 border border-[var(--dash-amber)] bg-[var(--dash-amber)]/10 text-[var(--dash-amber)] text-xs font-mono font-bold uppercase tracking-widest transition-all duration-150 active:scale-[0.97] cursor-pointer hover:bg-[var(--dash-amber)]/20 hover:shadow-[0_0_14px_rgba(245,158,11,0.2)]"
                >
                  Stall
                </button>
              </Tooltip>
              <Tooltip text="Hold Lead Train at next Station">
                <button
                  id="sim-hold-btn"
                  onClick={() => { holdLead() }}
                  className="w-full py-2 px-3 border border-[var(--dash-orange)] bg-[var(--dash-orange)]/10 text-[var(--dash-orange)] text-xs font-mono font-bold uppercase tracking-widest transition-all duration-150 active:scale-[0.97] cursor-pointer hover:bg-[var(--dash-orange)]/20 hover:shadow-[0_0_14px_rgba(249,115,22,0.2)]"
                >
                  Hold
                </button>
              </Tooltip>
            </div>
            <Tooltip text="Allow Lead Train to move" position="bottom">
              <button
                id="sim-release-btn"
                onClick={() => { releaseLead() }}
                className="w-full py-2 px-3 border border-[var(--dash-border-bright)] bg-transparent text-[var(--dash-text-bright)] text-xs font-mono uppercase tracking-widest transition-colors duration-150 cursor-pointer hover:border-[var(--dash-green)] hover:text-[var(--dash-green)]"
              >
                Release
              </button>
            </Tooltip>
          </div>
        )}

        {/* Spacer pushes button to bottom */}
        <div className="flex-1" />

        {/* Primary action */}
        <button
          id="sim-toggle-btn"
          onClick={handleToggle}
          className={`w-full py-3 px-4 border-2 font-bold tracking-[0.15em] uppercase text-sm font-mono transition-all duration-200 active:scale-[0.97] cursor-pointer ${running
            ? "border-[var(--dash-kill)] bg-[var(--dash-kill)]/10 text-[var(--dash-kill)] hover:bg-[var(--dash-kill)]/20 hover:shadow-[0_0_20px_var(--dash-kill-glow)]"
            : "border-[var(--dash-green)] bg-[var(--dash-green)]/10 text-[var(--dash-green)] hover:bg-[var(--dash-green)]/20 hover:shadow-[0_0_20px_rgba(34,197,94,0.25)]"
            }`}
        >
          {running ? "Stop Simulation" : "Start Simulation"}
        </button>
      </div>
    </PanelShell >
  );
}



