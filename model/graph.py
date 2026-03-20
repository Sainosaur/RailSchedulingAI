from dataclasses import dataclass

import networkx as net


@dataclass
class Station:
    """A Class to represent a station on Rail Line 104."""

    name: str
    position: int
    distance: float
    elevation: float
    passing_loop: bool


graph = net.Graph()

# Stations
chabowka = Station("Chabówka", 0, 0.582, 0.0, False)
rabka_zdroj = Station("Rabka-Zdrój", 1, 5.481, 0.0, False)
mszana_dolna = Station("Mszana Dolna", 2, 14.951, 0.0, False)
tymbark = Station("Tymbark", 3, 37.160, 0.0, False)
limanowa = Station("Limanowa", 4, 47.017, 0.0, False)
stary_sacz = Station("Marcinkowice", 5, 67.394, 0.0, False)
nowy_sacz = Station("Nowy Sącz", 6, 76.651, 0.0, False)
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
