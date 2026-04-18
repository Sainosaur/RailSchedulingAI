import asyncio
import websockets
import json

async def test_ws():
    uri = "ws://localhost:8000/ws/sim"
    async with websockets.connect(uri) as websocket:
        msg = await websocket.recv()
        data = json.loads(msg)
        print("Punctuality:", data.get("punctuality"))
        if "timetable" in data:
            print("Timetable Present: Yes")
        
asyncio.get_event_loop().run_until_complete(test_ws())
