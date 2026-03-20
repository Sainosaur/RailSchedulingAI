from dataclasses import dataclass
from operator import ne

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
    speed_limit: int | None


graph = net.Graph()

# Stations
chabowka = Station(
    "Chabówka", 0, 0.582, get_station_elevation("Dworzec Chabówka PKP"), True
)
rabka_zdroj = Station(
    "Rabka-Zdrój", 1, 5.481, get_station_elevation("Dworzec Rabka-Zdrój PKP"), False
)
mszana_dolna = Station(
    "Mszana Dolna", 2, 14.951, get_station_elevation("Przebudowa Mszana Dolna"), True
)
tymbark = Station(
    "Tymbark", 3, 37.160, get_station_elevation("Dworzec PKP Tymbark"), False
)
limanowa = Station(
    "Limanowa", 4, 47.017, get_station_elevation("Dworzec Limanowa PKP"), True
)
marcinkowice = Station(
    "Marcinkowice", 5, 67.394, get_station_elevation("Dworzec Marcinkowice PKP"), False
)
nowy_sacz = Station(
    "Nowy Sącz", 6, 76.651, get_station_elevation("Dworzec Nowy Sącz PKP"), True
)


# Turning stations into an array
stations = [
    chabowka,
    rabka_zdroj,
    mszana_dolna,
    tymbark,
    limanowa,
    marcinkowice,
    nowy_sacz,
]

# Adding stations and Segments to the graph
for station in stations:
    graph.add_node(station.name, data=station)
    limits = [90, 90, 60, 30, 30, 30]
    if stations.index(station) < len(stations) - 1:
        next_station = stations[(stations.index(station)) + 1]
        graph.add_edge(
            station.name,
            next_station.name,
            data=Segment(
                station,
                next_station,
                next_station.distance - station.distance,
                (next_station.elevation - station.elevation)
                / (next_station.distance - station.distance),
                limits[station.position],
            ),
        )
