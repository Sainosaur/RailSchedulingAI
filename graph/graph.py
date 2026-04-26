from __future__ import annotations
import sys
from dataclasses import dataclass, field
from pathlib import Path

_HEADWAY_TABLE: dict[int, tuple[float, float]] = {
    # index: (SH in metres, TH in seconds)
    0: (390.0, 15.6),
    1: (390.0, 15.6),
    2: (170.0, 10.2),
    3: (40.0, 4.8),
    4: (40.0, 4.8),
    5: (40.0, 4.8),
}


# Ensure project-root imports work regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# Also add the graph/ directory so 'utils' is found when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

import networkx as net

# Elevations (metres, SRTM30m via opentopodata) — hardcoded to avoid live
# HTTP calls on every import.  These are geographic constants for a fixed
# railway line and do not need to be fetched at runtime.
# TODO: Revert this hardcoding and restore the live REST API calls for elevation data
_STATION_ELEVATIONS: dict[str, float] = {
    "Dworzec Chabówka PKP":          553.0,
    "Dworzec Rabka-Zdrój PKP":       508.0,
    "Przebudowa Mszana Dolna":        363.0,
    "Dworzec PKP Tymbark":            312.0,
    "Dworzec Limanowa PKP":           313.0,
    "Dworzec Marcinkowice PKP":       292.0,
    "Dworzec Nowy Sącz PKP":          292.0,
}


@dataclass
class Station:
    """A Class to represent a station on Rail Line 104."""

    name: str
    position: int
    distance: float
    elevation: float
    passing_loop: bool


@dataclass
class Block:
    start: float
    end: float
    hazard: bool


@dataclass
class Segment:
    """A Class to represent a segment between two stations."""

    position: int
    start_station: Station
    end_station: Station
    distance: float
    gradient: float
    speed_limit: int | None
    hazard: bool


# ---------------------------------------------------------------------------
# Hardcoded SH and TH from internal_layers.md §3 (rounded values from doc)
# Key: segment index (0-based, matching station pair order)
# ---------------------------------------------------------------------------


@dataclass
class VLSegment:
    """
    A segment prepared for the Validation Layer.

    All values are in SI units (metres, m/s, seconds).
    Built once by build_vl_segments() and then shared with the VL.
    """

    id: str
    start: float  # metres along track
    end: float  # metres along track
    limit_ms: float  # speed limit  (m/s)
    spatial_headway: float  # SH — metres, no buffer
    temporal_headway: float  # TH — seconds, no buffer


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
        "Chabówka", 0, 0.582, _STATION_ELEVATIONS["Dworzec Chabówka PKP"], True
    )
    rabka_zdroj = Station(
        "Rabka-Zdrój", 1, 5.481, _STATION_ELEVATIONS["Dworzec Rabka-Zdrój PKP"], False
    )
    mszana_dolna = Station(
        "Mszana Dolna",
        2,
        14.951,
        _STATION_ELEVATIONS["Przebudowa Mszana Dolna"],
        True,
    )
    tymbark = Station(
        "Tymbark", 3, 37.160, _STATION_ELEVATIONS["Dworzec PKP Tymbark"], False
    )
    limanowa = Station(
        "Limanowa", 4, 47.017, _STATION_ELEVATIONS["Dworzec Limanowa PKP"], True
    )
    marcinkowice = Station(
        "Marcinkowice",
        5,
        67.394,
        _STATION_ELEVATIONS["Dworzec Marcinkowice PKP"],
        False,
    )
    nowy_sacz = Station(
        "Nowy Sącz", 6, 76.651, _STATION_ELEVATIONS["Dworzec Nowy Sącz PKP"], True
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
    limits = [90, 90, 60, 30, 30, 30]
    for i, station in enumerate(stations):
        g.add_node(station.name, data=station)

        if i < len(stations) - 1:
            next_station = stations[i + 1]
            start_m = round(station.distance * 1000)
            end_m = round(next_station.distance * 1000)
            sh, _ = _HEADWAY_TABLE[i]
            boundaries: list[Block] = []
            pos = float(start_m)
            while pos < end_m:
                boundaries.append(
                    Block(start=pos, end=min(pos + sh, float(end_m)), hazard=False)
                )
                pos += sh
            if boundaries[-1].end < end_m:
                boundaries.append(
                    Block(start=boundaries[-1].end, end=float(end_m), hazard=False)
                )
            # Merge the last zone into the second-to-last if it is shorter
            # than SH.  This guarantees every zone has length >= SH, which
            # is required for correct DTZ and headway spacing.
            if len(boundaries) >= 2:
                last_len = boundaries[-1].end - boundaries[-1].start
                if last_len < sh:
                    # Absorb last zone into its predecessor
                    boundaries[-2] = Block(
                        start=boundaries[-2].start,
                        end=boundaries[-1].end,
                        hazard=boundaries[-2].hazard or boundaries[-1].hazard,
                    )
                    boundaries.pop()
            g.add_edge(
                station.name,
                next_station.name,
                data=Segment(
                    i,
                    station,
                    next_station,
                    next_station.distance - station.distance,
                    (next_station.elevation - station.elevation)
                    / (next_station.distance - station.distance),
                    limits[i],
                    False,
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

    for _, (_, _, data) in enumerate(g.edges(data=True)):
        edge = data["data"]
        start_m = round(edge.start_station.distance * 1000)  # km → m
        end_m = round(edge.end_station.distance * 1000)
        limit_ms = round(edge.speed_limit / 3.6, 2)  # km/h → m/s

        sh, th = _HEADWAY_TABLE[edge.position]

        block_boundaries = []

        segments.append(
            VLSegment(
                id=edge.position,
                start=float(start_m),
                end=float(end_m),
                limit_ms=limit_ms,
                spatial_headway=sh,
                temporal_headway=th,
            )
        )

    _cached_vl_segments = segments
    return segments
