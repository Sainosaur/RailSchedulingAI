import { Button } from "@/components/ui/button";

const SEVERITY_STYLE = {
  critical:
    "text-[var(--dash-red)] border-[var(--dash-red)]/40 bg-[var(--dash-red)]/10",
  warning:
    "text-[var(--dash-orange)] border-[var(--dash-orange)]/40 bg-[var(--dash-orange)]/10",
  info: "text-[var(--dash-text)] border-[var(--dash-border-bright)] bg-white/5",
};

const STATUS_STYLE = {
  applied: "text-[var(--dash-green)] border-[var(--dash-green)]/40",
  pending: "text-[var(--dash-amber)] border-[var(--dash-amber)]/40",
  denied: "text-[var(--dash-red)] border-[var(--dash-red)]/40",
};

export default function RecommendationLog({ recommendations }) {
  return (
    <div className="h-full overflow-y-auto dash-scroll">
      {recommendations.map((rec) => (
        <div
          key={rec.id}
          className="flex flex-col gap-1 px-3 py-2.5 border-b border-[var(--dash-border)] hover:bg-white/[0.02] transition-colors"
        >
          <div className="flex items-center gap-2 flex-wrap">
            <span
              className={`text-[9px] font-mono font-semibold px-1.5 py-0.5 border uppercase tracking-wider ${SEVERITY_STYLE[rec.severity]}`}
            >
              {rec.severity}
            </span>
            <span className="text-[9px] font-mono px-1.5 py-0.5 border border-[var(--dash-border-bright)] text-[var(--dash-text-bright)] uppercase tracking-wider">
              {rec.station}
            </span>
            <span className="text-[9px] font-mono px-1.5 py-0.5 border border-[var(--dash-border-bright)] text-[var(--dash-text)] uppercase tracking-wider">
              {rec.type}
            </span>
            <span className="ml-auto text-[9px] font-mono text-[var(--dash-text)]/60">
              {rec.timestamp}
            </span>
          </div>
          <div className="flex items-start justify-between gap-2">
            <p className="text-[var(--dash-text-bright)] text-xs leading-relaxed flex-1">
              {rec.message}
            </p>
            <span
              className={`shrink-0 text-[9px] font-mono px-1.5 py-0.5 border uppercase tracking-wider ${STATUS_STYLE[rec.status]}`}
            >
              {rec.status}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}
