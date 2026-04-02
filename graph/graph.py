from dataclasses import dataclass, field

import sys
from pathlib import Path

# Ensure project-root imports work regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Also add the graph/ directory so 'utils' is found when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

import networkx as net

try:
    from graph.utils.station_detail import get_station_elevation
except ModuleNotFoundError:
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


# ---------------------------------------------------------------------------
# Hardcoded SH and TH from internal_layers.md §3 (rounded values from doc)
# Key: segment index (0-based, matching station pair order)
# ---------------------------------------------------------------------------
_HEADWAY_TABLE: dict[int, tuple[float, float]] = {
    # index: (SH in metres, TH in seconds)
    0: (312.5, 12.5),   # S0  90 km/h
    1: (312.5, 12.5),   # S1  90 km/h
    2: (138.9,  8.3),   # S2  60 km/h
    3: ( 34.7,  4.2),   # S3  30 km/h
    4: ( 34.7,  4.2),   # S4  30 km/h
    5: ( 34.7,  4.2),   # S5  30 km/h
}


@dataclass
class VLSegment:
    """
    A segment prepared for the Validation Layer.

    All values are in SI units (metres, m/s, seconds).
    Built once by build_vl_segments() and then shared with the VL.
    """
    id: str
    start: float              # metres along track
    end: float                # metres along track
    limit_ms: float           # speed limit  (m/s)
    spatial_headway: float    # SH — metres, no buffer
    temporal_headway: float   # TH — seconds, no buffer
    block_boundaries: list[float] = field(default_factory=list)
    # Ascending list of absolute positions (metres) where fixed blocks
    # begin within this segment.  The first boundary equals self.start;
    # subsequent boundaries are spaced SH apart.


# ---------------------------------------------------------------------------
# Module-level cache — computed once on first call, reused afterwards.
# ---------------------------------------------------------------------------
_cached_graph: net.Graph | None = None
_cached_vl_segments: list[VLSegment] | None = None


def graph() -> net.Graph:
    """
    Build (or return cached) NetworkX graph of Line 104.
    """
    global _cached_graph
    if _cached_graph is not None:
        return _cached_graph

    g = net.Graph()

    # Stations
    chabowka = Station(
        "Chabówka", 0, 0.582, get_station_elevation("Dworzec Chabówka PKP"), True
    )
    rabka_zdroj = Station(
        "Rabka-Zdrój", 1, 5.481, get_station_elevation("Dworzec Rabka-Zdrój PKP"), False
    )
    mszana_dolna = Station(
        "Mszana Dolna",
        2,
        14.951,
        get_station_elevation("Przebudowa Mszana Dolna"),
        True,
    )
    tymbark = Station(
        "Tymbark", 3, 37.160, get_station_elevation("Dworzec PKP Tymbark"), False
    )
    limanowa = Station(
        "Limanowa", 4, 47.017, get_station_elevation("Dworzec Limanowa PKP"), True
    )
    marcinkowice = Station(
        "Marcinkowice",
        5,
        67.394,
        get_station_elevation("Dworzec Marcinkowice PKP"),
        False,
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
        g.add_node(station.name, data=station)
        limits = [90, 90, 60, 30, 30, 30]
        if stations.index(station) < len(stations) - 1:
            next_station = stations[(stations.index(station)) + 1]
            g.add_edge(
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

    _cached_graph = g
    return g


def build_vl_segments() -> list[VLSegment]:
    """
    Build the list of VLSegment objects for the Validation Layer.

    * Converts graph.py distances (km) → metres, speed limits (km/h) → m/s.
    * Uses hardcoded SH and TH from internal_layers.md §3.
    * Divides each segment into fixed blocks of length SH and stores the
      block boundaries inside the VLSegment.
    * Result is cached — the conversion and lookup happen only once per run.
    """
    global _cached_vl_segments
    if _cached_vl_segments is not None:
        return _cached_vl_segments

    g = graph()
    segments: list[VLSegment] = []

    for i, (u, v_node, data) in enumerate(g.edges(data=True)):
        edge = data["data"]
        start_m = round(edge.start_station.distance * 1000)   # km → m
        end_m = round(edge.end_station.distance * 1000)
        limit_ms = round(edge.speed_limit / 3.6, 2)          # km/h → m/s

        sh, th = _HEADWAY_TABLE[i]

        # Build fixed-block boundaries within this segment.
        # Blocks start at segment start and are spaced SH apart.
        boundaries: list[float] = []
        pos = float(start_m)
        while pos < end_m:
            boundaries.append(pos)
            pos += sh
        # The segment end is always a boundary (may be a short final block)
        if boundaries[-1] < end_m:
            boundaries.append(float(end_m))

        segments.append(VLSegment(
            id=f"S{i}",
            start=float(start_m),
            end=float(end_m),
            limit_ms=limit_ms,
            spatial_headway=sh,
            temporal_headway=th,
            block_boundaries=boundaries,
        ))

    _cached_vl_segments = segments
    return segments
