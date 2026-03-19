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
