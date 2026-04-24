const SIGNAL = {
  green:         { color: '#22c55e', dots: 1 },
  amber:         { color: '#f59e0b', dots: 1 },
  'double-amber':{ color: '#f97316', dots: 2 },
  red:           { color: '#ef4444', dots: 1 },
}

export default function TrainNode({ data }) {
  const { trainId, signal, speed } = data
  const sig = SIGNAL[signal] ?? SIGNAL.green

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3 }}>
      {/* Signal dots */}
      <div style={{ display: 'flex', gap: 3, alignItems: 'center' }}>
        {Array.from({ length: sig.dots }).map((_, i) => (
          <span
            key={i}
            style={{
              width: 5,
              height: 5,
              borderRadius: '50%',
              background: sig.color,
              boxShadow: `0 0 4px ${sig.color}`,
              flexShrink: 0,
            }}
          />
        ))}
      </div>

      {/* Train body */}
      <div
        style={{
          padding: '2px 7px',
          border: `1px solid ${sig.color}`,
          background: `color-mix(in srgb, ${sig.color} 12%, #12151c)`,
          fontFamily: 'ui-monospace, Consolas, monospace',
          fontSize: 9,
          fontWeight: 600,
          color: sig.color,
          letterSpacing: '0.08em',
          whiteSpace: 'nowrap',
          lineHeight: 1.4,
        }}
      >
        {trainId}
      </div>

      {/* Speed */}
      <span
        style={{
          fontFamily: 'ui-monospace, Consolas, monospace',
          fontSize: 8,
          color: '#8892a4',
          letterSpacing: '0.05em',
          whiteSpace: 'nowrap',
        }}
      >
        {speed} km/h
      </span>
    </div>
  )
}
