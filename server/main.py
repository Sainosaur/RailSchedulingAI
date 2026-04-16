import csv
import os
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure project-root imports work regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from stable_baselines3.common.vec_env import VecNormalize

from graph.graph import graph
from validate_layer import log_manager
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


# Updates the graph with hazard data and notifies all WebSocket clients
@app.post("/api/hazard/{segmentPosition}/{block}/{status}")
async def hazard(segmentPosition: int, block: int, status: bool):
    nodes = [data["data"] for _, data in g.nodes(data=True)]
    start = next(s for s in nodes if s.position == segmentPosition)
    end = next(s for s in nodes if s.position == segmentPosition + 1)
    edge = g.get_edge_data(start.name, end.name)
    edge["data"].block_boundaries[block].hazard = status
    edge["data"].hazard = any(b.hazard for b in edge["data"].block_boundaries)
    await graph_manager.broadcast({"type": "graph_update", "graph": serialise_graph(g)})
    return {"segment": asdict(edge["data"]), "success": edge["data"].block_boundaries[block].hazard == status}


# Dashboard Endpoints
# REST
# Returns the graph to front end client
@app.get("/api/dashboard/graph")
async def root():
    return {"graph": serialise_graph(g)}


# Returns the logs to front end client
@app.get("/api/dashboard/logs")
async def log():
    path = log_manager.get_log_path()
    if not os.path.exists(path):
        return {"logs": []}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return {"logs": list(reader)}



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


# TODO
@app.get("/api/dashboard/recommendations")
async def recommendations():
    return {"recommendations": "Not Implemented"}


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
                
            lead_segment = raw_env.vl.get_segment(raw_env.lead_x)
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
                    "speed_ms": float(raw_env.lead_train_speed),
                    "speed_kmh": float(raw_env.lead_train_speed) * 3.6,
                    "signal": "green",
                    "segment": lead_segment.id,
                    "dwell_timer": 0,
                },
                "override": {
                    "active": False,
                    "proposed_a": 0.0,
                    "safe_a": 0.0,
                },
                "stations_visited": list(raw_env.visited_stations),
                "done": False,
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
    return {"status": "started"}


@app.post("/api/sim/stop")
async def stop_sim():
    await simulation_runner.stop()
    return {"status": "stopped"}
