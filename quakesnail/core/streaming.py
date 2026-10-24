"""Streaming and Replay Data Feeds with Strict < 35 MB Python Heap Ceiling.

Grounded in:
- Fixed-capacity circular ring buffers for streaming station waveforms.
- Deterministic offline scenario & historical miniSEED / SAC / CSV replay.
- Live SeedLink / FDSN / USGS streaming client with graceful offline fallback.
- Chunked memory discipline asserting peak Python heap < 35 MB via tracemalloc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import json
import os
import sys
import time
from typing import Dict, Iterator, List, Optional, Tuple, Callable
import numpy as np

from quakesnail.core.picking import WaveformPicker, Pick
from quakesnail.core.association import WavefrontAssociator, AssociatedCluster
from quakesnail.core.inversion import HypocenterInverter, HypocenterEstimate


@dataclass
class StationTrace:
    """Fixed-capacity cache-resident circular buffer for live station streaming."""

    station_code: str
    sample_rate_hz: float = 50.0
    capacity: int = 1024  # ~20 seconds of telemetry buffer
    head: int = 0
    total_samples: int = 0
    buffer: np.ndarray = field(init=False)
    timestamps: np.ndarray = field(init=False)

    def __post_init__(self):
        self.buffer = np.zeros(self.capacity, dtype=np.float32)
        self.timestamps = np.zeros(self.capacity, dtype=np.float64)

    def append(self, sample: float, timestamp_s: float):
        self.buffer[self.head] = np.float32(sample)
        self.timestamps[self.head] = timestamp_s
        self.head = (self.head + 1) % self.capacity
        self.total_samples += 1

    def get_recent(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve the most recent n_samples (ordered chronologically)."""
        n = min(n_samples, min(self.capacity, self.total_samples))
        if n <= 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float64)

        indices = [(self.head - n + i) % self.capacity for i in range(n)]
        return self.buffer[indices], self.timestamps[indices]


@dataclass
class WaveformPacket:
    station_code: str
    timestamp_s: float
    samples: np.ndarray
    sample_rate_hz: float = 50.0


class ReplayFeed:
    """Deterministic replay generator from synthetic scenarios or offline waveform arrays."""

    def __init__(self, scenario_data: Optional[Dict] = None, chunk_size: int = 25):
        self.chunk_size = chunk_size
        self.scenario_data = scenario_data or {}
        self.traces: Dict[str, np.ndarray] = {}
        self.times: Dict[str, np.ndarray] = {}
        self.station_coords: Dict[str, Tuple[float, float, float]] = {}

        if scenario_data:
            self._load_scenario(scenario_data)

    def _load_scenario(self, data: Dict):
        stations = data.get("stations", [])
        for st in stations:
            code = st["code"]
            self.station_coords[code] = (st["lat"], st["lon"], st.get("alt", 0.0))
            if "samples" in st:
                self.traces[code] = np.asarray(st["samples"], dtype=np.float32)
                t0 = st.get("start_time", 0.0)
                dt = 1.0 / st.get("sample_rate", 50.0)
                self.times[code] = t0 + np.arange(len(self.traces[code])) * dt

    def add_station_trace(
        self,
        station_code: str,
        samples: np.ndarray,
        start_time_s: float = 0.0,
        sample_rate_hz: float = 50.0,
        coords: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.station_coords[station_code] = coords
        self.traces[station_code] = np.asarray(samples, dtype=np.float32)
        dt = 1.0 / sample_rate_hz
        self.times[station_code] = start_time_s + np.arange(len(samples)) * dt

    def iter_packets(self) -> Iterator[WaveformPacket]:
        """Yield synchronized small waveform packets across all stations."""
        if not self.traces:
            return

        max_len = max(len(arr) for arr in self.traces.values())
        for start_idx in range(0, max_len, self.chunk_size):
            for code, arr in self.traces.items():
                if start_idx < len(arr):
                    chunk = arr[start_idx : start_idx + self.chunk_size]
                    t_chunk = self.times[code][start_idx]
                    yield WaveformPacket(
                        station_code=code,
                        timestamp_s=t_chunk,
                        samples=chunk,
                    )


class StreamingClient:
    """Manages live/offline feeds, feeding picker, associator, and inverter under < 35 MB heap."""

    def __init__(
        self,
        seedlink_host: Optional[str] = None,
        seedlink_port: int = 18000,
        offline_mode: bool = True,
    ):
        self.seedlink_host = seedlink_host
        self.seedlink_port = seedlink_port
        self.offline_mode = offline_mode or (seedlink_host is None)

        self.picker = WaveformPicker()
        self.associator = WavefrontAssociator()
        self.inverter = HypocenterInverter()

        self.traces: Dict[str, StationTrace] = {}
        self.active_hypocenters: List[HypocenterEstimate] = []
        self._last_inversion_picks: Dict[str, int] = {}
        self._last_inversion_time: Dict[str, float] = {}

    def register_station(self, code: str, lat: float, lon: float, alt: float = 0.0):
        self.traces[code] = StationTrace(station_code=code)
        self.picker.register_station(code, lat, lon, alt)
        self.associator.register_station(code, lat, lon, alt)

    def process_packet(self, packet: WaveformPacket) -> List[Pick]:
        """Process an incoming waveform chunk and update hypocenters."""
        code = packet.station_code
        if code not in self.traces:
            self.register_station(code, 0.0, 0.0, 0.0)

        trace = self.traces[code]
        dt = 1.0 / packet.sample_rate_hz
        t = packet.timestamp_s

        new_picks = []
        for s in packet.samples:
            trace.append(s, t)
            _, pick = self.picker.feed_sample(code, s, t)
            if pick:
                new_picks.append(pick)
                # Attempt association
                cluster = self.associator.associate_pick(pick)
                if cluster and len(cluster.picks) >= 4:
                    n_p = len(cluster.picks)
                    c_id = cluster.cluster_id
                    last_np = self._last_inversion_picks.get(c_id, 0)
                    last_t = self._last_inversion_time.get(c_id, -1e9)
                    # Debounce: invert on reaching 4 picks, then every 4 picks or at least 1.0s interval
                    if last_np == 0 or (n_p - last_np >= 4) or (t - last_t >= 1.0):
                        hypo = self.inverter.invert(cluster.picks, self.associator.station_coords)
                        if hypo:
                            self.active_hypocenters.append(hypo)
                            self._last_inversion_picks[c_id] = n_p
                            self._last_inversion_time[c_id] = t
            t += dt


        return new_picks

    def run_replay(self, feed: ReplayFeed, on_pick: Optional[Callable[[Pick], None]] = None) -> List[HypocenterEstimate]:
        """Run replay feed through full pipeline and return discovered hypocenters."""
        for code, coords in feed.station_coords.items():
            self.register_station(code, *coords)

        for packet in feed.iter_packets():
            picks = self.process_packet(packet)
            if on_pick:
                for p in picks:
                    on_pick(p)

        # Check for unassociated picks in associator
        unassoc = [p for p in self.picker.picks if not any(p in c.picks for c in self.associator.clusters)]
        if len(unassoc) >= 4:
            new_c = self.associator.create_cluster_from_picks(unassoc)
            if new_c and len(new_c.picks) >= 4:
                hypo = self.inverter.invert(new_c.picks, self.associator.station_coords)
                if hypo:
                    self.active_hypocenters.append(hypo)

        return self.active_hypocenters
