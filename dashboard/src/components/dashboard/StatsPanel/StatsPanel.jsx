import PanelShell from "@/components/layout/PanelShell";

export default function StatsPanel({ aiStats }) {
  if (!aiStats)
    return (
      <PanelShell title="Live AI Stats" className="h-full">
        <div className="flex items-center justify-center h-full text-[var(--dash-text)]/40 text-[10px] uppercase font-mono tracking-widest">
          Waiting for Data...
        </div>
      </PanelShell>
    );

  const {
    speed_ms,
    speed_kmh,
    acceleration,
    approaching_station,
    dist_to_next_station,
    dist_to_obstruction,
    signal,
    authority_ranges,
    segment_id,
  } = aiStats;

  return (
    <PanelShell title="Live AI Stats" className="h-full">
      <div className="p-3 flex flex-col gap-4 h-full overflow-y-auto dash-scroll">
        {/* Speed & Acceleration Grid */}
        <div className="grid grid-cols-2 gap-2">
          <StatBox
            label="Speed"
            value={`${speed_kmh.toFixed(1)} km/h`}
            subValue={`${speed_ms.toFixed(2)} m/s`}
          />
          <StatBox label="Accel" value={`${acceleration.toFixed(2)} m/s²`} />
          <StatBox
            label="Dist to Occupied"
            value={`${(dist_to_obstruction / 1000).toFixed(2)} km`}
            subValue={`${dist_to_obstruction.toFixed(0)} m`}
          />
          <SignalBox label="Signal" aspect={signal} />
        </div>

        {/* Navigation Info */}
        <div className="space-y-2">
          <div className="flex justify-between items-end border-b border-[var(--dash-border)] pb-1">
            <span className="text-[10px] text-[var(--dash-text)] uppercase font-mono">
              Next Station
            </span>
            <span className="text-xs text-[var(--dash-text-bright)] font-semibold">
              {approaching_station}
            </span>
          </div>
          <div className="flex justify-between items-end border-b border-[var(--dash-border)] pb-1">
            <span className="text-[10px] text-[var(--dash-text)] uppercase font-mono">
              Dist to Station
            </span>
            <span className="text-xs text-[var(--dash-text-bright)]">
              {(dist_to_next_station / 1000).toFixed(2)} km
            </span>
          </div>
          <div className="flex justify-between items-end border-b border-[var(--dash-border)] pb-1">
            <span className="text-[10px] text-[var(--dash-text)] uppercase font-mono">
              Segment ID
            </span>
            <span className="text-xs text-[var(--dash-text-bright)] font-mono">
              {segment_id}
            </span>
          </div>
        </div>

        {/* Reward Info */}
        {aiStats.reward && (
          <div className="space-y-2">
            <div className="flex justify-between items-end border-b border-[var(--dash-border)] pb-1">
              <span className="text-[10px] text-[var(--dash-text)] uppercase font-mono">
                Step Reward
              </span>
              <span className={`text-xs font-bold font-mono ${aiStats.reward.total >= 0 ? "text-[var(--dash-green)]" : "text-[var(--dash-red)]"}`}>
                {aiStats.reward.total.toFixed(2)}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1">
              {Object.entries(aiStats.reward.breakdown || {}).map(([k, v]) => (
                <div key={k} className="flex justify-between text-[9px] font-mono">
                  <span className="text-[var(--dash-text)]/60">{k.replace("r_", "")}</span>
                  <span className={v >= 0 ? "text-[var(--dash-green)]/80" : "text-[var(--dash-red)]/80"}>
                    {v.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Authority Ranges */}
        {authority_ranges && (
          <div className="flex flex-col gap-1.5 mt-auto">
            <span className="text-[10px] text-[var(--dash-text)] uppercase font-mono tracking-wider">
              Authority Ranges (SH)
            </span>
            <div className="flex h-1.5 w-full bg-[var(--dash-border)] rounded-full overflow-hidden">
              <div
                style={{ width: "25%" }}
                className="bg-[var(--dash-red)] h-full"
              />
              <div
                style={{ width: "25%" }}
                className="bg-[var(--dash-orange)] h-full"
              />
              <div
                style={{ width: "25%" }}
                className="bg-[var(--dash-amber)] h-full"
              />
              <div
                style={{ width: "25%" }}
                className="bg-[var(--dash-green)] h-full"
              />
            </div>
            <div className="grid grid-cols-4 text-[8px] font-mono text-[var(--dash-text)]/60 text-center uppercase">
              <span>1.0x</span>
              <span>2.0x</span>
              <span>3.0x</span>
              <span>Max</span>
            </div>
          </div>
        )}
      </div>
    </PanelShell>
  );
}

function StatBox({ label, value, subValue }) {
  return (
    <div className="bg-white/[0.03] border border-[var(--dash-border)] p-2 flex flex-col gap-0.5">
      <span className="text-[9px] text-[var(--dash-text)] uppercase font-mono">
        {label}
      </span>
      <span className="text-sm text-[var(--dash-text-head)] font-bold tracking-tight">
        {value}
      </span>
      {subValue && (
        <span className="text-[9px] text-[var(--dash-text)]/50 font-mono">
          {subValue}
        </span>
      )}
    </div>
  );
}

function SignalBox({ label, aspect }) {
  const colors = {
    green: "bg-[var(--dash-green)]",
    amber: "bg-[var(--dash-orange)]",
    "double-amber": "bg-[var(--dash-amber)]",
    red: "bg-[var(--dash-red)]",
  };

  return (
    <div className="bg-white/[0.03] border border-[var(--dash-border)] p-2 flex flex-col gap-2">
      <span className="text-[9px] text-[var(--dash-text)] uppercase font-mono">
        {label}
      </span>
      <div className="flex items-center gap-2">
        <div
          className={`w-3 h-3 rounded-full ${colors[aspect] || "bg-gray-500"} shadow-[0_0_8px_rgba(255,255,255,0.2)]`}
        />
        <span className="text-[10px] text-[var(--dash-text-head)] font-bold uppercase tracking-widest font-mono">
          {aspect || "Unknown"}
        </span>
      </div>
    </div>
  );
}
