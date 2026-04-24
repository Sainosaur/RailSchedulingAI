import PanelShell from "@/components/layout/PanelShell";
import { Play, Square, RotateCcw } from "lucide-react";
import { startSimulation, stopSimulation, resetSimulation } from "@/services/simulation";

export default function StopwatchPanel({ simTime, isRunning, onSimChange }) {
  const formatTime = (seconds) => {
    const totalSeconds = Math.floor(seconds || 0);
    const h = Math.floor(totalSeconds / 3600);
    const m = Math.floor((totalSeconds % 3600) / 60);
    const s = totalSeconds % 60;
    return `${h.toString().padStart(2, "0")}:${m
      .toString()
      .padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  };

  const handleStart = async () => {
    await startSimulation();
    if (onSimChange) onSimChange(true);
  };

  const handleStop = async () => {
    await stopSimulation();
    if (onSimChange) onSimChange(false);
  };

  const handleReset = async () => {
    await resetSimulation();
    if (onSimChange) onSimChange(false);
  };

  return (
    <PanelShell title="Simulation Controls" className="mt-auto">
      <div className="flex flex-col items-center justify-center py-4 gap-4">
        {/* Stopwatch Display */}
        <div className="text-4xl font-mono text-[var(--dash-text-bright)] tracking-widest bg-black/30 px-6 py-2 rounded-lg border border-[var(--dash-border)] shadow-inner">
          {formatTime(simTime)}
        </div>

        {/* Control Buttons */}
        <div className="flex gap-4">
          {!isRunning ? (
            <button
              onClick={handleStart}
              className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-md transition-colors text-xs font-semibold uppercase tracking-wider"
            >
              <Play size={14} fill="currentColor" />
              Start
            </button>
          ) : (
            <button
              onClick={handleStop}
              className="flex items-center gap-2 px-4 py-2 bg-amber-600 hover:bg-amber-500 text-white rounded-md transition-colors text-xs font-semibold uppercase tracking-wider"
            >
              <Square size={14} fill="currentColor" />
              End Sim
            </button>
          )}

          <button
            onClick={handleReset}
            className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-md transition-colors text-xs font-semibold uppercase tracking-wider"
          >
            <RotateCcw size={14} />
            Reset
          </button>
        </div>
      </div>
    </PanelShell>
  );
}
