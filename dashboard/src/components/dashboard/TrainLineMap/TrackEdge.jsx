import { getStraightPath } from "@xyflow/react";

export default function TrackEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  data,
}) {
  const [edgePath] = getStraightPath({ sourceX, sourceY, targetX, targetY });

  const hazard = data?.hazard;
  const stroke = hazard ? "#dc2626" : "#2a3045";
  const strokeDash = hazard ? undefined : "6 4";
  const strokeWidth = hazard ? 2.5 : 1.5;

  return (
    <g>
      <path
        id={id}
        d={edgePath}
        fill="none"
        stroke={stroke}
        strokeWidth={strokeWidth}
        strokeDasharray={strokeDash}
        strokeLinecap="round"
      />
    </g>
  );
}
