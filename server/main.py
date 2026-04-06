import csv
import os

from fastapi import FastAPI, websockets
from fastapi.middleware.cors import CORSMiddleware

from graph import graph
from validate_layer import log_manager

app = FastAPI()
graph = graph()
STATUS = "Running"  # Temporary placeholder for /kill endpoints
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


# Returns the graph to front end client
@app.get("/api/dashboard/graph")
async def root():
    return {"graph": graph}


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
    global STATUS
    STATUS = "Killed"
    return {"status": STATUS}


# TODO
@app.post("/api/dashboard/kill/restore")
async def restore_system():
    global STATUS
    STATUS = "Running"
    return {"status": STATUS}


# TODO
@app.get("/api/dashboard/kill")
async def system_status():
    return {"status": STATUS}


# TODO
@app.get("/api/dashboard/recommendations")
async def recommendations():
    return {"recommendations": "Not Implemented"}
