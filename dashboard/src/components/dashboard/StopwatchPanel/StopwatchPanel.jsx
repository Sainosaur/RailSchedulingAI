import PanelShell from "@/components/layout/PanelShell";
import { RotateCcw } from "lucide-react";
import { resetSimulation } from "@/services/simulation";

export default function StopwatchPanel({ simTime, onSimChange }) {
  const formatTime = (seconds) => {
    const totalSeconds = Math.floor(seconds || 0);
    const h = Math.floor(totalSeconds / 3600);
    const m = Math.floor((totalSeconds % 3600) / 60);
    const s = totalSeconds % 60;
    return `${h.toString().padStart(2, "0")}:${m
      .toString()
      .padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  const handleReset = async () => {
    await resetSimulation();
    if (onSimChange) onSimChange(false);
  };

  return (
    <PanelShell 
      title="Watch" 
      className="mt-auto"
      actions={
        <button
          onClick={handleReset}
          className="text-[var(--dash-text)]/50 hover:text-[var(--dash-text-bright)] transition-colors p-1"
          title="Reset Simulation"
        >
          <RotateCcw size={14} />
        </button>
      }
    >
      <div className="flex flex-col items-center justify-center py-6">
        {/* Stopwatch Display */}
        <div className="text-4xl font-mono text-[var(--dash-text-bright)] tracking-widest bg-black/20 px-8 py-3 rounded-lg border border-[var(--dash-border)]/50 shadow-inner">
          {formatTime(simTime)}
        </div>
      </div>
    </PanelShell>
  );
}
