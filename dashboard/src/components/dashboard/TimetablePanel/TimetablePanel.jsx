import PanelShell from "@/components/layout/PanelShell";

export default function TimetablePanel({ timetable, punctuality }) {
  const formatTime = (seconds) => {
    if (seconds === undefined || seconds === null) return "--:--";
    const m = Math.floor(Math.abs(seconds) / 60)
      .toString()
      .padStart(2, "0");
    const s = Math.floor(Math.abs(seconds) % 60)
      .toString()
      .padStart(2, "0");
    const sign = seconds < 0 ? "-" : "";
    return `${sign}${m}:${s}`;
  };

  const getStatusColor = (status) => {
    if (status === "late") return "text-[var(--dash-red)] font-semibold";
    if (status === "early") return "text-emerald-400 font-semibold";
    if (status === "on_time") return "text-[var(--dash-text-bright)]";
    return "text-[var(--dash-text)]/50";
  };

  const formatStatus = (statusLabel, diff) => {
    if (statusLabel === "unknown") return "-";
    if (statusLabel === "on_time") return "On Time";
    const absDiff = Math.abs(diff);
    const m = Math.floor(absDiff / 60);
    const s = Math.floor(absDiff % 60);
    const formattedDiff = m > 0 ? `${m}m ${s}s` : `${s}s`;

    if (statusLabel === "late") return `Late (${formattedDiff})`;
    if (statusLabel === "early") return `Early (${formattedDiff})`;
    return "On Time";
  };

  return (
    <PanelShell title="Timetable" className="h-full">
      <div className="h-full overflow-y-auto dash-scroll px-3 py-2">
        <table className="w-full text-left text-[11px] font-mono whitespace-nowrap">
          <thead className="text-[var(--dash-text)]/50 sticky top-0 bg-[#121212] uppercase border-b border-[var(--dash-border)]">
            <tr>
              <th className="font-normal py-1 pr-2">Station</th>
              <th className="font-normal py-1 pr-2 text-right">Sch Arr</th>
              <th className="font-normal py-1 pr-2 text-right">Act/ETA</th>
              <th className="font-normal py-1 text-right">Status</th>
            </tr>
          </thead>
          <tbody className="text-[var(--dash-text-bright)]">
            {!timetable || timetable.length === 0 || timetable.error ? (
              <tr>
                <td colSpan={4} className="text-center py-4">
                  Timetable cannot be found! Ensure a simulation is currently
                  running
                </td>
              </tr>
            ) : (
              timetable.map((entry) => {
                const isPast =
                  typeof punctuality?.ai?.arrival_log?.[entry.station_idx] !==
                  "undefined";
                const isNext =
                  punctuality?.ai?.next_station_idx === entry.station_idx;

                let actualTime = null;
                let statusLabel = "unknown";
                let timeDiff = 0;

                if (isPast) {
                  actualTime = punctuality.ai.arrival_log[entry.station_idx];
                  timeDiff = entry.scheduled_arrival - actualTime; // positive = early, negative = late
                  if (Math.abs(timeDiff) <= 60) statusLabel = "on_time";
                  else if (timeDiff > 0) statusLabel = "early";
                  else statusLabel = "late";
                } else if (isNext) {
                  actualTime = punctuality.ai.eta;
                  statusLabel = punctuality.ai.status;
                  timeDiff = punctuality.ai.slack_seconds;
                }

                return (
                  <tr
                    key={entry.station_idx}
                    className={`border-b border-[var(--dash-border)]/30 last:border-0 ${isNext ? "bg-[var(--dash-text)]/5" : ""}`}
                  >
                    <td className="py-1.5 pr-2 truncate max-w-[100px] text-[var(--dash-text)]">
                      {entry.station_name}
                    </td>
                    <td className="py-1.5 pr-2 text-right">
                      {formatTime(entry.scheduled_arrival)}
                    </td>
                    <td className="py-1.5 pr-2 text-right text-[var(--dash-text-bright)]">
                      {formatTime(actualTime)}
                    </td>
                    <td
                      className={`py-1.5 text-right ${getStatusColor(statusLabel)}`}
                    >
                      {formatStatus(statusLabel, timeDiff)}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </PanelShell>
  );
}
