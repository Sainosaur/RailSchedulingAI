import csv
import os
import sys
from dataclasses import asdict
from pathlib import Path
import asyncio

# Ensure project-root imports work regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from stable_baselines3.common.vec_env import VecNormalize

from graph.graph import graph
from validate_layer import log_manager as vl_log_manager
from server.simulation import SimulationRunner


def serialise_graph(g) -> dict:
    """Convert NetworkX graph to a JSON-serialisable dict."""
    nodes = [asdict(data["data"]) for _, data in g.nodes(data=True)]
    edges = [
        {
            "source": u,
            "target": v,
            **asdict(data["data"]),
        }
        for u, v, data in g.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges}


class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, data: dict):
        for ws in self.active:
            await ws.send_json(data)


app = FastAPI()
g = graph()
graph_manager = ConnectionManager()
sim_manager = ConnectionManager()
simulation_runner = SimulationRunner(broadcast_callback=sim_manager.broadcast)
lead_status_manager = ConnectionManager()
log_manager = ConnectionManager()

KILLED = False  # Temporary placeholder for /kill endpoints
origins = [
    "http://localhost:5173",
]


def _get_raw_env():
    env = simulation_runner.venv.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def watch_logs():
    last_mtime = 0
    last_size = 0
    while True:
        log_path = vl_log_manager.get_log_path()
        if os.path.exists(log_path):
            stat = os.stat(log_path)
            if stat.st_size != last_size or stat.st_mtime != last_mtime:
                last_size = stat.st_size
                last_mtime = stat.st_mtime
                
                try:
                    with open(log_path, "r", newline="") as fh:
                        reader = csv.DictReader(fh)
                        logs = list(reader)
                        if logs:
                            await log_manager.broadcast({"type": "log_update", "logs": logs})
                except Exception as e:
                    print(f"Error reading logs: {e}")
        await asyncio.sleep(0.5)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(watch_logs())



# Updates the graph with hazard data and notifies all WebSocket clients
@app.post("/api/hazard/{segmentPosition}/{block}/{status}")
async def hazard(segmentPosition: int, block: int, status: bool):
    nodes = [data["data"] for _, data in g.nodes(data=True)]
    start = next(s for s in nodes if s.position == segmentPosition)
    end = next(s for s in nodes if s.position == segmentPosition + 1)
    edge = g.get_edge_data(start.name, end.name)
    edge["data"].block_boundaries[block].hazard = status
    edge["data"].hazard = any(b.hazard for b in edge["data"].block_boundaries)
    # Propagate hazard to the running simulation environment
    if simulation_runner.venv is not None:
        raw_env = _get_raw_env()
        block_obj = edge["data"].block_boundaries[block]
        raw_env.set_block_hazard(block_obj.start, block_obj.end, status)
    await graph_manager.broadcast({"type": "graph_update", "graph": serialise_graph(g)})
    return {"segment": asdict(edge["data"]), "success": edge["data"].block_boundaries[block].hazard == status}


# Dashboard Endpoints
# REST
# Returns the graph to front end client
@app.get("/api/dashboard/graph")
async def root():
    return {"graph": serialise_graph(g)}


# Returns the logs to front end client
@app.websocket("/ws/logs")
async def log_updates(websocket: WebSocket):
    await log_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep connection alive
    except WebSocketDisconnect:
        log_manager.disconnect(websocket)


# BUG 15 FIX: Wire up the kill/restore endpoints directly to the
# simulation runner instead of just flipping a cosmetic 'KILLED' flag.
@app.post("/api/dashboard/kill/kill")
async def kill_system():
    global KILLED
    KILLED = True
    await simulation_runner.stop()
    return {"killed": KILLED, "reason": "Simulation stopped by kill switch."}


@app.post("/api/dashboard/kill/restore")
async def restore_system():
    global KILLED
    KILLED = False
    await simulation_runner.start()
    return {"killed": KILLED, "reason": "Simulation restored."}


@app.get("/api/dashboard/kill")
async def system_status():
    return {"killed": KILLED, "reason": ""}


# Lead train controls
async def _broadcast_lead_status():
    """Push lead train state to all /ws/lead/status clients."""
    raw_env = _get_raw_env()
    await lead_status_manager.broadcast({
        "stalled": raw_env.lead_stalled,
        "held": raw_env.lead_held,
        "speed_ms": float(raw_env.lead_v),
        "dwell_timer": int(raw_env.lead_dwell_timer),
    })


@app.post("/api/sim/lead/stall")
async def stall_lead():
    """Force the lead train to emergency brake and stop."""
    if simulation_runner.venv is None:
        return {"error": "Simulation not loaded", "stalled": False}
    raw_env = _get_raw_env()
    raw_env.stall_lead()
    await _broadcast_lead_status()
    return {"stalled": True}


@app.post("/api/sim/lead/release")
async def release_lead():
    """Release the lead train stall/hold."""
    if simulation_runner.venv is None:
        return {"error": "Simulation not loaded", "stalled": False}
    raw_env = _get_raw_env()
    raw_env.release_lead()
    await _broadcast_lead_status()
    return {"stalled": False}


@app.post("/api/sim/lead/hold")
async def hold_lead():
    """Hold the lead train at its current station."""
    if simulation_runner.venv is None:
        return {"error": "Simulation not loaded", "held": False}
    raw_env = _get_raw_env()
    raw_env.hold_lead()
    await _broadcast_lead_status()
    return {"held": True}

@app.websocket("/ws/lead/status")
async def lead_status(websocket: WebSocket):
    await lead_status_manager.connect(websocket)
    
    if simulation_runner.venv is not None:
        raw_env = _get_raw_env()
        await websocket.send_json({
            "stalled": raw_env.lead_stalled,
            "held": raw_env.lead_held,
        })
    else:
        await websocket.send_json(None)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        lead_status_manager.disconnect(websocket)

# TODO
@app.get("/api/dashboard/recommendations")
async def recommendations():
    return {"recommendations": "Not Implemented"}

@app.get("/api/sim/timetable")
async def get_timetable():
    """Return the currently loaded timetable."""
    if simulation_runner.venv is None:
        return {"error": "Simulation not loaded", "timetable": []}
    raw_env = _get_raw_env()
    return raw_env.timetable.to_dict()


# Web Sockets
@app.websocket("/ws/graph")
async def graph_updates(websocket: WebSocket):
    await graph_manager.connect(websocket)
    await websocket.send_json({"type": "graph_update", "graph": serialise_graph(g)})
    try:
        while True:
            await websocket.receive_text()  # keep connection alive
    except WebSocketDisconnect:
        graph_manager.disconnect(websocket)


@app.websocket("/ws/sim")
async def sim_updates(websocket: WebSocket):
    await sim_manager.connect(websocket)
    try:
        if simulation_runner.venv is not None:
            raw_env = _get_raw_env()
            segment = raw_env.vl.get_segment(raw_env.x)
            progress = (raw_env.x - segment.start) / (segment.end - segment.start)
            next_st_idx = min(raw_env.last_station_idx + 1, len(raw_env.STATIONS) - 1)
            next_st_pos = raw_env.STATIONS[next_st_idx]
            headway = float((raw_env.lead_x - raw_env.x) / raw_env.v if raw_env.v > 0.01 else 9999.0)
            signal_aspect = raw_env._get_signal_aspect()
            if signal_aspect == 3:
                 signal = "green"
            elif signal_aspect == 2:
                signal = "double-amber"
            elif signal_aspect == 1:
                signal = "amber"
            else:
                signal = "red"
                
            lead_segment = raw_env.vl.get_segment(min(raw_env.lead_x, raw_env.TRACK_END))
            lead_progress = (raw_env.lead_x - lead_segment.start) / (lead_segment.end - lead_segment.start)
            
            await websocket.send_json({
                "type": "sim_update",
                "step": raw_env.step_count,
                "time": raw_env.time,
                "ai": {
                    "progress": float(progress),
                    "speed_ms": float(raw_env.v),
                    "speed_kmh": float(raw_env.v) * 3.6,
                    "dtz": float(raw_env.dtz),
                    "signal": signal,
                    "dist_to_next_station": float(next_st_pos - raw_env.x),
                    "speed_limit_ms": float(segment.limit_ms),
                    "headway": headway,
                    "segment": segment.id,
                },
                "lead": {
                    "progress": float(lead_progress),
                    "speed_ms": float(raw_env.lead_v),
                    "speed_kmh": float(raw_env.lead_v) * 3.6,
                    "signal": "green",
                    "segment": lead_segment.id,
                    "dwell_timer": int(raw_env.lead_dwell_timer),
                    "stalled": raw_env.lead_stalled,
                    "held": raw_env.lead_held,
                },
                "override": {
                    "active": False,
                    "proposed_a": 0.0,
                    "safe_a": 0.0,
                },
                "stations_visited": list(raw_env.visited_stations),
                "hazards": [{"start": s, "end": e} for s, e in raw_env.active_hazards],
                "done": False,
                "timetable": raw_env.timetable.to_dict(),
                "punctuality": raw_env.get_punctuality_status(),
            })
            await asyncio.sleep(0.5)
        while True:
            await websocket.receive_text()  # keep connection alive
    except WebSocketDisconnect:
        sim_manager.disconnect(websocket)


# Simulation Endpoints
@app.post("/api/sim/start")
async def start_sim(lead_speed: float = 20.0):
    if simulation_runner.model is None or simulation_runner.venv is None:
        simulation_runner.load_model(lead_train_speed=lead_speed)
    await simulation_runner.reset()
    await simulation_runner.start()
    await _broadcast_lead_status()
    return {"status": "started"}


@app.post("/api/sim/stop")
async def stop_sim():
    await simulation_runner.stop()
    return {"status": "stopped"}

@app.get("/api/sim/status")
async def sim_status():
    return {"status": simulation_runner.is_running}