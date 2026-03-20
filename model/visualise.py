import matplotlib.pyplot as plt
import networkx as net

from graph import graph, stations

# Position nodes horizontally by distance, vertically by elevation
pos = {s.name: (s.distance, s.elevation) for s in stations}

edge_labels = {
    (u, v): f"{d['data'].distance:.1f}km\n{d['data'].speed_limit}km/h"
    for u, v, d in graph.edges(data=True)
}

node_colors = ["lightgreen" if s.passing_loop else "lightblue" for s in stations]

plt.figure(figsize=(14, 6))
net.draw(
    graph,
    pos=pos,
    with_labels=True,
    node_color=node_colors,
    node_size=2000,
    font_size=8,
)
net.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels, font_size=7)

plt.title("Rail Line 104: Chabówka → Nowy Sącz")
plt.xlabel("Distance (km)")
plt.ylabel("Elevation (m)")
plt.tight_layout()
plt.show()
