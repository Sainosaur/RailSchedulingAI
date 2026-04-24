import PanelShell from "@/components/layout/PanelShell";
import { useState, useEffect, useRef } from "react";

const SIGNAL_COLOR = {
  green: "#22c55e",
  amber: "#f59e0b",
  "double-amber": "#f97316",
  red: "#ef4444",
};

// Station positions in metres (must match backend STATIONS array)
const STATION_POSITIONS_M = [582, 5481, 14951, 37160, 47017, 67394, 76651];

function lerp(a, b, t) {
  return a + t * (b - a);
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

export default function TrainLineMap({ trains }) {
  const [stations, setStations] = useState([]);
  const [segments, setSegments] = useState([]);
  const wsRef = useRef(null);

  // Connect to graph WebSocket ONCE via useEffect — prevents the listener
  // duplication bug that caused trains to glitch on every re-render.
  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/graph");
    wsRef.current = ws;

    ws.addEventListener("open", () => {
      console.log("Backend connected (Graph)");
    });

    const handleMessage = (event) => {
      const data = JSON.parse(event.data);
      setStations(data.graph.nodes);
      setSegments(Object.values(data.graph.edges));
    };

    ws.addEventListener("message", handleMessage);

    return () => {
      ws.removeEventListener("message", handleMessage);
      ws.close();
    };
  }, []);

  // Build a lookup from station position index → station data for train rendering
  const stationByPosition = Object.fromEntries(
    stations.map((s) => [s.position, s]),
  );

  const legend = (
    <div className="flex items-center gap-4 text-[10px] font-mono text-[var(--dash-text)] uppercase tracking-widest">
      <span className="flex items-center gap-1.5">
        <span className="inline-block w-4 h-px border-t border-dashed border-[#2a3045]" />
        Track
      </span>
      <span className="flex items-center gap-1.5">
        <span className="inline-block w-4 h-[2.5px] bg-[var(--dash-kill)]" />
        Hazard
      </span>
      {[
        { signal: "green", color: "#22c55e", label: "Green", dots: 1 },
        { signal: "amber", color: "#f59e0b", label: "Amber", dots: 1 },
        {
          signal: "double-amber",
          color: "#f97316",
          label: "Dbl Amber",
          dots: 2,
        },
        { signal: "red", color: "#ef4444", label: "Red", dots: 1 },
      ].map(({ signal, color, label, dots }) => (
        <span key={signal} className="flex items-center gap-1">
          {Array.from({ length: dots }).map((_, i) => (
            <span
              key={i}
              className="inline-block w-2 h-2 rounded-full"
              style={{ background: color, boxShadow: `0 0 4px ${color}` }}
            />
          ))}
          <span>{label}</span>
        </span>
      ))}
    </div>
  );

  return (
    <PanelShell
      title="AI Simulation Control and Observation Panel"
      badge="LINE 104"
      actions={legend}
      className="w-full h-full"
    >
      <div style={{ width: "100%", height: "100%", padding: "12px" }}>
        <svg
          viewBox="0 0 960 190"
          style={{ width: "100%", height: "100%", overflow: "visible" }}
        >
          <defs>
            <filter id="sig-glow" x="-80%" y="-80%" width="260%" height="260%">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Track segments with block boundaries */}
          {segments.map((seg) => {
            const src = seg.start_station;
            const tgt = seg.end_station;
            if (!src || !tgt) return null;

            const srcX = src.distance * 12 + 20,
              srcY = src.elevation / 10;
            const tgtX = tgt.distance * 12 + 20,
              tgtY = tgt.elevation / 10;

            if (!seg.block_boundaries?.length) {
              return (
                <line
                  key={src.position}
                  x1={srcX}
                  y1={srcY}
                  x2={tgtX}
                  y2={tgtY}
                  stroke="#2a3045"
                  strokeWidth={1.5}
                  strokeDasharray="6 4"
                  strokeLinecap="round"
                />
              );
            }

            const segStartM = src.distance * 1000;
            const segEndM = tgt.distance * 1000;

            const distToSVG = (d) => {
              const t = (d - segStartM) / (segEndM - segStartM);
              return { x: lerp(srcX, tgtX, t), y: lerp(srcY, tgtY, t) };
            };

            const dx = tgtX - srcX,
              dy = tgtY - srcY;
            const len = Math.sqrt(dx * dx + dy * dy);
            const nx = len > 0 ? -dy / len : 0;
            const ny = len > 0 ? dx / len : 1;
            const TICK = 7;

            return (
              <g key={src.position}>
                {seg.block_boundaries.map((block, i) => {
                  const p1 = distToSVG(block.start);
                  const p2 = distToSVG(block.end);
                  const hazard = seg.hazard || block.hazard;
                  return (
                    <line
                      key={i}
                      x1={p1.x}
                      y1={p1.y}
                      x2={p2.x}
                      y2={p2.y}
                      stroke={hazard ? "#dc2626" : "#2a3045"}
                      strokeWidth={1}
                      strokeDasharray={hazard ? undefined : "6 4"}
                      strokeLinecap="round"
                    />
                  );
                })}
                {seg.block_boundaries.map((block, i) => {
                  if (i === 0) return null;
                  const p = distToSVG(block.start);
                  const isHazard =
                    block.hazard || seg.block_boundaries[i - 1]?.hazard;
                  return (
                    <line
                      key={`t${i}`}
                      x1={p.x - nx * TICK}
                      y1={p.y - ny * TICK}
                      x2={p.x + nx * TICK}
                      y2={p.y + ny * TICK}
                      stroke={isHazard ? "#dc2626" : "#3a4055"}
                      strokeWidth={isHazard ? 2 : 0.05}
                    />
                  );
                })}
              </g>
            );
          })}

          {/* Stations */}
          {stations.map((s) => (
            <g key={s.position}>
              <circle
                cx={s.distance * 12 + 20}
                cy={s.elevation / 10}
                r={5}
                fill="#c8d0de"
                stroke="#2a3045"
                strokeWidth={1.5}
              />
              <text
                x={s.distance * 12 + 20}
                y={s.elevation / 10 + 17}
                textAnchor="middle"
                fill="#c8d0de"
                fontSize={9}
                fontFamily="ui-monospace, Consolas, monospace"
                letterSpacing="0.1em"
                style={{ userSelect: "none" }}
              >
                {s.name}
              </text>
            </g>
          ))}

          {/* Trains — position using absolute position_m when available,
              falling back to segment + progress for backward compat. */}
          {trains.map((train, idx) => {
            let x, y;

            if (train.position_m != null && stations.length > 0) {
              // Use absolute position to compute SVG coordinates directly.
              // Find which segment the train is in by position.
              const pos = train.position_m;
              const totalStart = STATION_POSITIONS_M[0];
              const totalEnd =
                STATION_POSITIONS_M[STATION_POSITIONS_M.length - 1];
              // Find the two bounding stations by absolute position
              let sIdx = 0;
              for (let i = 0; i < STATION_POSITIONS_M.length - 1; i++) {
                if (pos >= STATION_POSITIONS_M[i]) sIdx = i;
              }
              const s0 = stationByPosition[sIdx];
              const s1 = stationByPosition[sIdx + 1];
              if (s0 && s1) {
                const segStart = STATION_POSITIONS_M[sIdx];
                const segEnd = STATION_POSITIONS_M[sIdx + 1];
                const t = clamp((pos - segStart) / (segEnd - segStart), 0, 1);
                const sx0 = s0.distance * 12 + 20,
                  sy0 = s0.elevation / 10;
                const sx1 = s1.distance * 12 + 20,
                  sy1 = s1.elevation / 10;
                x = lerp(sx0, sx1, t);
                y = lerp(sy0, sy1, t);
              } else {
                // Fallback: use global linear interpolation
                const t = clamp(
                  (pos - totalStart) / (totalEnd - totalStart),
                  0,
                  1,
                );
                const firstSt = stations[0] || {
                  distance: 0.582,
                  elevation: 553,
                };
                const lastSt = stations[stations.length - 1] || {
                  distance: 76.651,
                  elevation: 292,
                };
                x = lerp(
                  firstSt.distance * 12 + 20,
                  lastSt.distance * 12 + 20,
                  t,
                );
                y = lerp(firstSt.elevation / 10, lastSt.elevation / 10, t);
              }
            } else {
              // Legacy: use segment index + progress fraction
              const seg = segments[train.segment];
              if (!seg) return null;
              const src = stations[train.segment];
              const tgt = stations[train.segment + 1];
              if (!src || !tgt) return null;
              const progress = clamp(train.progress ?? 0, 0, 1);
              x = lerp(
                src.distance * 12 + 20,
                tgt.distance * 12 + 20,
                progress,
              );
              y = lerp(src.elevation / 10, tgt.elevation / 10, progress);
            }

            const color = SIGNAL_COLOR[train.signal] ?? "#22c55e";
            const isDoubleAmber = train.signal === "double-amber";
            const label = train.id || (idx === 0 ? "LEAD" : "AI");

            return (
              <g key={idx} transform={`translate(${x}, ${y})`}>
                {/* Train label above track */}
                <rect
                  x={-17}
                  y={-28}
                  width={34}
                  height={13}
                  fill={color}
                  fillOpacity={0.12}
                  stroke={color}
                  strokeWidth={1}
                  rx={1}
                />
                <text
                  x={0}
                  y={-18}
                  textAnchor="middle"
                  fill={color}
                  fontSize={8}
                  fontWeight={700}
                  fontFamily="ui-monospace, Consolas, monospace"
                  letterSpacing="0.08em"
                  style={{ userSelect: "none" }}
                >
                  {label}
                </text>

                {/* Stem from label to signal dot */}
                <line
                  x1={0}
                  y1={-15}
                  x2={0}
                  y2={-4}
                  stroke={color}
                  strokeWidth={1}
                  strokeOpacity={0.35}
                />

                {/* Signal dot(s) ON the track */}
                {isDoubleAmber ? (
                  <>
                    <circle
                      cx={-4}
                      cy={0}
                      r={3}
                      fill={color}
                      filter="url(#sig-glow)"
                    />
                    <circle
                      cx={4}
                      cy={0}
                      r={3}
                      fill={color}
                      filter="url(#sig-glow)"
                    />
                  </>
                ) : (
                  <circle
                    cx={0}
                    cy={0}
                    r={3}
                    fill={color}
                    filter="url(#sig-glow)"
                  />
                )}

                {/* Speed below */}
                <text
                  x={0}
                  y={6}
                  textAnchor="middle"
                  fill="#8892a4"
                  fontSize={7}
                  fontFamily="ui-monospace, Consolas, monospace"
                  letterSpacing="0.04em"
                  style={{ userSelect: "none" }}
                >
                  {train.speed_kmh !== undefined
                    ? Math.round(train.speed_kmh)
                    : 0}{" "}
                  km/h
                </text>

                {/* Distance to nearest obstruction (AI train only) */}
                {train.dist_to_obstruction != null && (
                  <text
                    x={0}
                    y={15}
                    textAnchor="middle"
                    fill="#6b7280"
                    fontSize={6}
                    fontFamily="ui-monospace, Consolas, monospace"
                    letterSpacing="0.04em"
                    style={{ userSelect: "none" }}
                  >
                    d_occ: {Math.round(train.dist_to_obstruction)}m
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      </div>
    </PanelShell>
  );
}
