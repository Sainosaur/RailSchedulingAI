"""
environment/timetable.py

Timetable data model and schedule generator for Line 104.

Generates a per-station timetable at episode start using segment speed
limits, acceleration/deceleration buffers, and fixed dwell times.
Provides live ETA and punctuality status for both AI and lead trains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TimetableEntry:
    """One row of the timetable — a single station stop."""

    station_idx: int
    station_name: str
    position_m: float
    scheduled_arrival: float  # seconds from episode start
    scheduled_dwell: float  # seconds (0 for origin and terminus)
    scheduled_departure: float  # arrival + dwell

    def to_dict(self) -> dict:
        return {
            "station_idx": self.station_idx,
            "station_name": self.station_name,
            "position_m": self.position_m,
            "scheduled_arrival": round(self.scheduled_arrival, 1),
            "scheduled_dwell": round(self.scheduled_dwell, 1),
            "scheduled_departure": round(self.scheduled_departure, 1),
        }


@dataclass
class Timetable:
    """Full timetable for one episode — a list of TimetableEntry objects."""

    entries: list[TimetableEntry] = field(default_factory=list)

    def to_dict(self) -> list[dict]:
        """JSON-serialisable list for WebSocket broadcast."""
        return [e.to_dict() for e in self.entries]

    def get_entry(self, station_idx: int) -> Optional[TimetableEntry]:
        """Lookup a station by index. Returns None if not found."""
        for e in self.entries:
            if e.station_idx == station_idx:
                return e
        return None

    def next_entry_after(self, station_idx: int) -> Optional[TimetableEntry]:
        """Return the timetable entry for the station after the given index."""
        for e in self.entries:
            if e.station_idx > station_idx:
                return e
        return None


# ---------------------------------------------------------------------------
# ETA helper — segment-aware with accel/decel buffer
# ---------------------------------------------------------------------------


def _compute_segment_travel_time(
    distance_m: float,
    speed_limit_ms: float,
    accel: float = 0.5,
    decel: float = 0.5,
) -> float:
    # Ensure accel/decel are non-zero to avoid NaNs
    accel = max(0.01, accel)
    decel = max(0.01, decel)
    speed_limit_ms = max(0.1, speed_limit_ms)
    distance_m = max(0.0, distance_m)

    if distance_m == 0:
        return 0.0

    # Distance required to accelerate to limit
    d_accel = (speed_limit_ms**2) / (2 * accel)
    # Distance required to decelerate from limit
    d_decel = (speed_limit_ms**2) / (2 * decel)

    if d_accel + d_decel >= distance_m:
        # Triangular profile — never reaches full speed
        # Peak velocity: v_peak² = 2 * distance * (accel * decel) / (accel + decel)
        v_peak = math.sqrt(2 * distance_m * (accel * decel) / (accel + decel))
        t_accel = v_peak / accel
        t_decel = v_peak / decel
        return t_accel + t_decel
    else:
        # Trapezoidal profile
        t_accel = speed_limit_ms / accel
        t_decel = speed_limit_ms / decel
        d_cruise = distance_m - d_accel - d_decel
        t_cruise = d_cruise / speed_limit_ms
        return t_accel + t_cruise + t_decel


def compute_eta_to_station(
    current_position: float,
    target_position: float,
    segments: list,
    accel: float = 0.5,
    decel: float = 0.5,
) -> float:
    """Estimate time from current_position to target_position using
    segment speed limits with acceleration/deceleration buffers.

    This computes the time for each segment portion the train still
    needs to traverse, applying a trapezoidal profile for the current
    segment (accounts for needing to slow down for the target) and
    raw segment-limit traversal for intermediate segments.

    Parameters
    ----------
    current_position : float  Current position in metres.
    target_position  : float  Target station position in metres.
    segments         : list   VLSegment objects from the ValidationLayer.
    accel            : float  Traction acceleration (m/s²).
    decel            : float  Service braking deceleration magnitude (m/s²).

    Returns
    -------
    float  Estimated travel time in seconds.
    """
    if target_position <= current_position:
        return 0.0

    remaining = target_position - current_position
    total_time = 0.0

    for seg in segments:
        if seg.end <= current_position:
            continue
        if seg.start >= target_position:
            break

        # Portion of this segment the train must traverse
        seg_start = max(seg.start, current_position)
        seg_end = min(seg.end, target_position)

        if seg_end <= seg_start:
            continue

        seg_dist = seg_end - seg_start

        # Use segment speed limit with accel/decel buffer
        total_time += _compute_segment_travel_time(seg_dist, seg.limit_ms, accel, decel)

    return total_time


# ---------------------------------------------------------------------------
# Timetable generator
# ---------------------------------------------------------------------------

# Station names on Line 104 (matching STATIONS position array in railway_env)
STATION_NAMES: list[str] = [
    "Chabówka",
    "Rabka-Zdrój",
    "Mszana Dolna",
    "Tymbark",
    "Limanowa",
    "Marcinkowice",
    "Nowy Sącz",
]


def generate_timetable(
    segments: list,
    station_positions: list[float],
    dwell_seconds: float = 120.0,
    slack_factor: float = 1.10,  # 10% operational buffer
    station_names: list[str] | None = None,
    accel: float = 0.5,
    decel: float = 0.5,
) -> Timetable:
    """Generate a timetable for one episode.

    Computes scheduled arrival times at each station using segment speed
    limits with acceleration/deceleration buffers, plus fixed dwell at
    intermediate stations.

    Parameters
    ----------
    segments          : list of VLSegment objects from the Validation Layer.
    station_positions : list of station positions in metres (ascending).
    dwell_seconds     : float  Scheduled dwell time at intermediate stations.
    slack_factor      : float  Multiplier for travel times (1.10 = 10% slack).
    station_names     : optional list of human-readable station names.
                        Falls back to STATION_NAMES if not provided.
    accel             : float  Traction acceleration (m/s²).
    decel             : float  Service braking deceleration magnitude (m/s²).

    Returns
    -------
    Timetable  Populated timetable with one entry per station.
    """
    names = station_names or STATION_NAMES
    if len(names) < len(station_positions):
        # Pad with generic names if needed
        names = list(names) + [
            f"Station {i}" for i in range(len(names), len(station_positions))
        ]

    entries: list[TimetableEntry] = []
    cumulative_time = 0.0

    for i, pos in enumerate(station_positions):
        if i == 0:
            # Origin station — departure at t=0, no dwell
            entries.append(
                TimetableEntry(
                    station_idx=i,
                    station_name=names[i],
                    position_m=pos,
                    scheduled_arrival=0.0,
                    scheduled_dwell=0.0,
                    scheduled_departure=0.0,
                )
            )
            continue

        # Travel time from previous station to this one
        prev_pos = station_positions[i - 1]
        travel_time = compute_eta_to_station(prev_pos, pos, segments, accel, decel)

        # Apply slack to the travel time portion
        travel_time *= slack_factor

        # Add departure dwell from previous station (except origin handled above)
        if i > 1:
            cumulative_time += dwell_seconds

        cumulative_time += travel_time

        # Terminus gets no dwell; intermediate stations get scheduled dwell
        is_terminus = i == len(station_positions) - 1
        dwell = 0.0 if is_terminus else dwell_seconds

        entries.append(
            TimetableEntry(
                station_idx=i,
                station_name=names[i],
                position_m=pos,
                scheduled_arrival=cumulative_time,
                scheduled_dwell=dwell,
                scheduled_departure=cumulative_time + dwell,
            )
        )
    return Timetable(entries=entries)
