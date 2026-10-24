"""
Dynamic Real-Time P- and S-Wave Arrival Countdown Engine.

Computes theoretical P-wave and S-wave travel times and dynamic real-time
countdowns (seconds_p, seconds_s) for affected municipalities.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from quakesnail.spatial.catalog import MunicipalityIntensity


# Standard default crustal / upper-mantle velocities (km/s)
DEFAULT_VP_CRUST = 6.0
DEFAULT_VS_CRUST = 3.5
DEFAULT_VP_MANTLE = 8.0
DEFAULT_VS_MANTLE = 4.6
DEFAULT_MOHO_DEPTH_KM = 35.0


@dataclass(frozen=True, slots=True)
class ArrivalCountdownResult:
    """Real-time seismic wave arrival countdown state for a specific location."""
    distance_hypo_km: float
    distance_epi_km: float
    travel_time_p: float
    travel_time_s: float
    seconds_p: float
    seconds_s: float
    p_arrived: bool
    s_arrived: bool
    p_countdown_str: str
    s_countdown_str: str
    status: str  # "PENDING", "P_ARRIVED_S_PENDING", "S_ARRIVED"


class ArrivalCountdownEngine:
    """
    Computes theoretical P and S travel times and dynamic real-time arrival countdowns.
    """

    @classmethod
    def compute_travel_times(
        cls,
        distance_hypo_km: float,
        distance_epi_km: Optional[float] = None,
        depth_km: Optional[float] = None,
        vp: Optional[float] = None,
        vs: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Compute theoretical P-wave and S-wave travel times (in seconds).

        Uses smooth 1D velocity profile with Moho refraction (Pn / Sn) for regional
        distances when distance_epi_km is provided, or direct hypo ray travel time.

        Parameters
        ----------
        distance_hypo_km : float
            Hypocentral distance in kilometers.
        distance_epi_km : Optional[float]
            Epicentral surface distance in kilometers.
        depth_km : Optional[float]
            Focal depth in kilometers.
        vp : Optional[float]
            Custom P-wave velocity in km/s (if None, uses model).
        vs : Optional[float]
            Custom S-wave velocity in km/s (if None, uses model).

        Returns
        -------
        Tuple[float, float]
            (travel_time_p, travel_time_s) in seconds.
        """
        if vp is not None and vs is not None:
            return distance_hypo_km / vp, distance_hypo_km / vs

        d_hypo = max(distance_hypo_km, 0.1)
        d_epi = distance_epi_km if distance_epi_km is not None else d_hypo

        # Smooth effective apparent velocity transition from crustal to upper-mantle refraction
        # Near-field (<100 km): Vp ~ 5.9 km/s, Vs ~ 3.45 km/s
        # Regional (>300 km): Vp ~ 7.8 km/s, Vs ~ 4.45 km/s
        trans = d_epi / (d_epi + 140.0)
        eff_vp = DEFAULT_VP_CRUST + (DEFAULT_VP_MANTLE - DEFAULT_VP_CRUST) * trans * 0.95
        eff_vs = DEFAULT_VS_CRUST + (DEFAULT_VS_MANTLE - DEFAULT_VS_CRUST) * trans * 0.95

        t_p = d_hypo / eff_vp
        t_s = d_hypo / eff_vs

        return max(t_p, 0.01), max(t_s, 0.02)

    @classmethod
    def compute_countdowns(
        cls,
        origin_time: float,
        distance_hypo_km: float,
        distance_epi_km: Optional[float] = None,
        depth_km: Optional[float] = None,
        current_time: Optional[float] = None,
    ) -> ArrivalCountdownResult:
        """
        Compute real-time arrival countdowns relative to current time.

        Parameters
        ----------
        origin_time : float
            Earthquake origin time in Unix epoch seconds.
        distance_hypo_km : float
            Hypocentral distance in km.
        distance_epi_km : Optional[float]
            Epicentral distance in km.
        depth_km : Optional[float]
            Focal depth in km.
        current_time : Optional[float]
            Current time in Unix epoch seconds (defaults to time.time()).

        Returns
        -------
        ArrivalCountdownResult
        """
        if current_time is None:
            current_time = time.time()

        d_epi = distance_epi_km if distance_epi_km is not None else distance_hypo_km
        t_p, t_s = cls.compute_travel_times(
            distance_hypo_km=distance_hypo_km,
            distance_epi_km=d_epi,
            depth_km=depth_km,
        )

        arrival_p = origin_time + t_p
        arrival_s = origin_time + t_s

        seconds_p = arrival_p - current_time
        seconds_s = arrival_s - current_time

        p_arrived = seconds_p <= 0.0
        s_arrived = seconds_s <= 0.0

        p_countdown_str = cls.format_countdown(seconds_p)
        s_countdown_str = cls.format_countdown(seconds_s)

        if not p_arrived:
            status = "PENDING"
        elif not s_arrived:
            status = "P_ARRIVED_S_PENDING"
        else:
            status = "S_ARRIVED"

        return ArrivalCountdownResult(
            distance_hypo_km=distance_hypo_km,
            distance_epi_km=d_epi,
            travel_time_p=t_p,
            travel_time_s=t_s,
            seconds_p=round(seconds_p, 2),
            seconds_s=round(seconds_s, 2),
            p_arrived=p_arrived,
            s_arrived=s_arrived,
            p_countdown_str=p_countdown_str,
            s_countdown_str=s_countdown_str,
            status=status,
        )

    @staticmethod
    def format_countdown(seconds_remaining: float) -> str:
        """
        Format countdown seconds into user-facing status string.

        Positive: e.g. "14.2s"
        Zero/Negative: e.g. "Arrived (T+5s)"
        """
        if seconds_remaining > 0.0:
            return f"{seconds_remaining:.1f}s"
        else:
            elapsed = abs(seconds_remaining)
            return f"Arrived (T+{elapsed:.0f}s)"

    @classmethod
    def batch_update_countdowns(
        cls,
        municipalities: List[MunicipalityIntensity],
        origin_time: float,
        current_time: Optional[float] = None,
    ) -> List[MunicipalityIntensity]:
        """
        Update seconds_p and seconds_s for a list of MunicipalityIntensity objects
        in real time without mutating original objects.
        """
        from quakesnail.spatial.catalog import MunicipalityIntensity

        if current_time is None:
            current_time = time.time()

        updated: List[MunicipalityIntensity] = []
        for m in municipalities:
            if getattr(m, "travel_time_p", 0.0) > 0.0:
                t_p = m.travel_time_p
                t_s = m.travel_time_s
            else:
                t_p, t_s = cls.compute_travel_times(distance_hypo_km=m.distance_km)
            sec_p = origin_time + t_p - current_time
            sec_s = origin_time + t_s - current_time

            updated.append(
                MunicipalityIntensity(
                    name=m.name,
                    country_code=m.country_code,
                    lat=m.lat,
                    lon=m.lon,
                    distance_km=m.distance_km,
                    pga_g=m.pga_g,
                    pgv_cms=m.pgv_cms,
                    mmi=m.mmi,
                    jma_shindo=m.jma_shindo,
                    seconds_p=round(sec_p, 2),
                    seconds_s=round(sec_s, 2),
                    pga_gal=m.pga_gal,
                    mmi_roman=m.mmi_roman,
                    azimuth_deg=m.azimuth_deg,
                    tier=m.tier,
                    region=m.region,
                    travel_time_p=t_p,
                    travel_time_s=t_s,
                )
            )

        return updated
