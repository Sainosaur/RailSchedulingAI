from dataclasses import dataclass

import networkx as net
from utils.station_detail import get_station_elevation


@dataclass
class Station:
    """A Class to represent a station on Rail Line 104."""

    name: str
    position: int
    distance: float
    elevation: float
    passing_loop: bool


@dataclass
class Segment:
    """A Class to represent a segment between two stations."""

    start_station: Station
    end_station: Station
    distance: float
    gradient: float
    speed_limit: int


graph = net.Graph()

# Stations
chabowka = Station("Chabówka", 0, 0.582, get_station_elevation("Chabówka"), False)
rabka_zdroj = Station(
    "Rabka-Zdrój", 1, 5.481, get_station_elevation("Rabka-Zdrój"), False
)
mszana_dolna = Station(
    "Mszana Dolna", 2, 14.951, get_station_elevation("Mszana Dolna"), False
)
tymbark = Station("Tymbark", 3, 37.160, get_station_elevation("Tymbark PKP"), False)
limanowa = Station("Limanowa", 4, 47.017, get_station_elevation("Limanowa"), False)
stary_sacz = Station(
    "Marcinkowice", 5, 67.394, get_station_elevation("Marcinkowice"), False
)
nowy_sacz = Station("Nowy Sącz", 6, 76.651, get_station_elevation("Nowy Sącz"), False)


# Turning stations into an array
stations = [
    chabowka,
    rabka_zdroj,
    mszana_dolna,
    tymbark,
    limanowa,
    stary_sacz,
    nowy_sacz,
]

# Adding stations to the graph
for station in stations:
    graph.add_node(station.name, data=station)
    if station.elevation == None:
        print(station)
