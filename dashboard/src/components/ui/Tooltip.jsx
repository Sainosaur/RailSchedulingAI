import { useState, useRef, useEffect } from "react";

/**
 * Custom tooltip that matches the dashboard's dark industrial aesthetic.
 * Wraps any child element and shows a styled tooltip on hover.
 *
 * @param {string}  text       – Tooltip text
 * @param {"top"|"bottom"}  position – Where to place the tooltip (default: top)
 * @param {React.ReactNode} children – The trigger element
 */
export default function Tooltip({ text, position = "top", children }) {
  const [visible, setVisible] = useState(false);
  const timeoutRef = useRef(null);

  function show() {
    clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(() => setVisible(true), 300);
  }

  function hide() {
    clearTimeout(timeoutRef.current);
    setVisible(false);
  }

  useEffect(() => () => clearTimeout(timeoutRef.current), []);

  const positionClasses =
    position === "bottom"
      ? "top-full mt-2 left-1/2 -translate-x-1/2"
      : "bottom-full mb-2 left-1/2 -translate-x-1/2";

  const caretClasses =
    position === "bottom"
      ? "bottom-full left-1/2 -translate-x-1/2 border-b-[var(--dash-border-bright)] border-t-transparent border-x-transparent"
      : "top-full left-1/2 -translate-x-1/2 border-t-[var(--dash-border-bright)] border-b-transparent border-x-transparent";

  return (
    <div className="relative" onMouseEnter={show} onMouseLeave={hide}>
      {children}
      {visible && (
        <div
          className={`absolute z-50 pointer-events-none ${positionClasses}`}
          style={{ whiteSpace: "nowrap" }}
        >
          <div
            className="px-2.5 py-1.5 border border-[var(--dash-border-bright)] bg-[var(--dash-bg)] text-[var(--dash-text-bright)] text-[10px] font-mono tracking-wide shadow-[0_4px_20px_rgba(0,0,0,0.5)] animate-in fade-in duration-150"
          >
            {text}
          </div>
          {/* Caret */}
          <div
            className={`absolute w-0 h-0 border-[4px] ${caretClasses}`}
          />
        </div>
      )}
    </div>
  );
}
