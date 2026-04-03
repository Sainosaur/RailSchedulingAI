import csv

from fastapi import FastAPI, websockets

from graph import graph

app = FastAPI()

graph = graph()


@app.get("/api/dashboard/graph")
async def root():
    return {"graph": graph}


@app.get("/api/dashboard/logs")
async def log():
    logs = csv.reader("../validate_layer/override_log.csv")
    return {"logs": logs}


@app.get("/api/dashboard/trains")
async def trains():
    return {"trains": "Not Implemented"}


@app.post("/api/dashboard/kill/kill")
async def kill_system():

    return {"status": "Killed"}


@app.post("/api/dashboard/kill/restore")
async def restore_system():
    return {"status": "Running"}


@app.get("/api/dashboard/kill")
async def system_status():
    return {"status": "Running"}


@app.get("/api/dashboard/recommendations")
async def recommendations():
    return {"recommendations": "Not Implemented"}
