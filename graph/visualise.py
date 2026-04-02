import sys
from pathlib import Path

# Ensure project-root imports work regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import networkx as net

from graph import graph as build_graph

# Build the graph and extract station data from nodes
g = build_graph()
stations = [data["data"] for _, data in g.nodes(data=True)]

# Position nodes horizontally by distance, vertically by elevation
pos = {s.name: (s.distance, s.elevation) for s in stations}

edge_labels = {
    (u, v): f"{d['data'].distance:.1f}km\n{d['data'].speed_limit}km/h"
    for u, v, d in g.edges(data=True)
}

node_colors = ["lightgreen" if s.passing_loop else "lightblue" for s in stations]

plt.figure(figsize=(14, 6))
net.draw(
    g,
    pos=pos,
    with_labels=True,
    node_color=node_colors,
    node_size=2000,
    font_size=8,
)
net.draw_networkx_edge_labels(g, pos=pos, edge_labels=edge_labels, font_size=7)

plt.title("Rail Line 104: Chabówka → Nowy Sącz")
plt.xlabel("Distance (km)")
plt.ylabel("Elevation (m)")
plt.tight_layout()
plt.show()
