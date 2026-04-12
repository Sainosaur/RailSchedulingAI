import csv
import os
from dataclasses import asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

try:
    from graph.graph import graph
except ImportError:
    from graph import graph
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
manager = ConnectionManager()
simulation_runner = SimulationRunner(broadcast_callback=manager.broadcast)

KILLED = False  # Temporary placeholder for /kill endpoints
origins = [
    "http://localhost:5173",
]

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
    await manager.broadcast({"type": "graph_update", "graph": serialise_graph(g)})
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


# TODO
@app.get("/api/dashboard/trains")
async def trains():
    return {"trains": "Not Implemented"}


# TODO
@app.post("/api/dashboard/kill/kill")
async def kill_system():
    global KILLED
    KILLED = True
    return {"killed": KILLED, "reason": "Not Implemented"}


# TODO
@app.post("/api/dashboard/kill/restore")
async def restore_system():
    global KILLED
    KILLED = False
    return {"killed": KILLED, "reason": "Not Implemented"}


# TODO
@app.get("/api/dashboard/kill")
async def system_status():
    return {"killed": KILLED, "reason": "Not Implemented"}


# TODO
@app.get("/api/dashboard/recommendations")
async def recommendations():
    return {"recommendations": "Not Implemented"}


# Web Sockets
@app.websocket("/ws/graph")
async def graph_updates(websocket: WebSocket):
    await manager.connect(websocket)
    await websocket.send_json({"type": "graph_update", "graph": serialise_graph(g)})
    try:
        while True:
            await websocket.receive_text()  # keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.websocket("/ws/sim")
async def sim_updates(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)


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
