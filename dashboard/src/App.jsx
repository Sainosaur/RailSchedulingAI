import { useState, useEffect } from "react";
import "./App.css";
import { INITIAL_KILL_SWITCH } from "@/data/railData";

import {
  killServer,
  restoreServer,
  getKillStatus,
} from "@/services/killSwitch";
import axios from "axios";
import TrainLineMap from "@/components/dashboard/TrainLineMap/TrainLineMap";
import KillSwitchPanel from "@/components/dashboard/KillSwitchPanel/KillSwitchPanel";
import OverrideLogPanel from "@/components/dashboard/OverrideLogPanel/OverrideLogPanel";
import StatsPanel from "@/components/dashboard/StatsPanel/StatsPanel";
import TimetablePanel from "@/components/dashboard/TimetablePanel/TimetablePanel";
import SimulationControlPanel from "@/components/dashboard/SimulationControlPanel/SimulationControlPanel";

const simSocket = new WebSocket("ws://localhost:8000/ws/sim");

export default function App() {
  const [killState, setKillState] = useState(INITIAL_KILL_SWITCH);
  const [logs, setLogs] = useState([]);
  const [aiStats, setAiStats] = useState(null);
  const [trains, setTrains] = useState([]);
  const [timetable, setTimetable] = useState(null);
  const [punctuality, setPunctuality] = useState(null);
  const [leadState, setLeadState] = useState({ stalled: false, held: false });

  useEffect(() => {
    simSocket.onopen = () => console.log("Backend connected (Simulation)");

    const handleMessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "sim_update") {
        setTrains([data.lead, data.ai]);
        setAiStats(data.ai);
        setTimetable(data.timetable);
        setPunctuality(data.punctuality);
        setLeadState({ stalled: data.lead.stalled, held: data.lead.held });
      }
    };

    simSocket.addEventListener("message", handleMessage);

    getKillStatus().then((status) => {
      setKillState(status);
    });

    // Poll for logs every 2 seconds
    const logInterval = setInterval(() => {
      axios
        .get("http://localhost:8000/api/dashboard/logs")
        .then((res) => {
          const fetchedLogs = res.data.logs.map((l, index) => ({
            id: `L-${l.timestamp}-${index}`,
            timestamp: new Date(
              parseFloat(l.timestamp) * 1000,
            ).toLocaleTimeString(),
            level: "SAFETY",
            source: "VLS",
            message: `Override: ${l.original_ppo_a} -> ${l.corrected_a} (${l.constraint_id})`,
          }));
          setLogs(fetchedLogs);
        })
        .catch((err) => console.error("Log fetch failed", err));
    }, 2000);

    return () => {
      simSocket.removeEventListener("message", handleMessage);
      clearInterval(logInterval);
    };
  }, []);

  const handleKill = () => {
    killServer()
      .then(() => {
        setKillState((prev) => ({
          ...prev,
          killed: true,
        }));
      })
      .catch(() => {
        console.error("Kill failed");
      });
  };

  function handleRestore() {
    restoreServer()
      .then(() => {
        setKillState((prev) => ({
          ...prev,
          killed: false,
        }));
      })
      .catch(() => {
        console.error("Restore failed");
      });
  }

  function handleReasonChange(reason) {
    setKillState((prev) => ({ ...prev, reason }));
  }

  function handleResetLogs() {
    axios
      .post("http://localhost:8000/api/dashboard/logs/reset")
      .then(() => {
        setLogs([]);
      })
      .catch((err) => console.error("Log reset failed", err));
  }

  return (
    <div className="w-screen h-screen grid grid-rows-[2fr_3fr] gap-2 p-2 overflow-hidden">
      {/* Row 1: Train line map */}
      <TrainLineMap trains={trains} />

      {/* Row 2: Bottom panels organized for better space usage */}
      <div className="grid grid-cols-[280px_1fr_1fr_1fr] gap-2 min-h-0">
        {/* Controls Column (Stacked Vertically) */}
        <div className="grid grid-rows-[auto_1fr] gap-2 min-h-0">
          <KillSwitchPanel
            killed={killState.killed}
            reason={killState.reason}
            onKill={handleKill}
            onRestore={handleRestore}
            onReasonChange={handleReasonChange}
          />
          <SimulationControlPanel
            stalled={leadState.stalled}
            held={leadState.held}
          />
        </div>

        <OverrideLogPanel logs={logs} onReset={handleResetLogs} />
        <StatsPanel aiStats={aiStats} />
        <TimetablePanel
          timetable={timetable || undefined}
          punctuality={punctuality || undefined}
        />
      </div>
    </div>
  );
}
