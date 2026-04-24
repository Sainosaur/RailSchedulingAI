import PanelShell from "@/components/layout/PanelShell";
import RecommendationLog from "./RecommendationLog";

export default function RecommendationsPanel({ recommendations }) {
  return (
    <PanelShell title="Recommendation Log" className="h-full">
      <div className="grid divide-x divide-[var(--dash-border)] h-full">
        {/* Left: Recommendation log */}
        <div className="flex flex-col overflow-hidden">
          <div className="px-3 py-1.5 border-b border-[var(--dash-border)] bg-[var(--dash-bg)]/50 shrink-0">
            <span className="text-[9px] font-mono text-[var(--dash-text)] uppercase tracking-widest">
              Log
            </span>
          </div>
          <div className="flex-1 overflow-hidden">
            <RecommendationLog recommendations={recommendations} />
          </div>
        </div>
      </div>
    </PanelShell>
  );
}
