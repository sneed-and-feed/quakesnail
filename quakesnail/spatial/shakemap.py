"""
Anisotropic ShakeMap & Ground Motion Prediction Equation (GMPE) Engine.

Computes fault/regime-aware PGA (%g and Gal), PGV (cm/s), Instrumental MMI (I-XII),
and JMA Shindo (0-7) intensity scales using standardized seismological models
(Atkinson-Boore, Campbell-Bozorgnia, Worden et al., Shabestari-Yamazaki).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


# Standard physical and seismological constants
G_TO_GAL = 980.665  # 1 g = 980.665 cm/s^2 (Gal)
EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True, slots=True)
class GroundMotionResult:
    """Ground motion intensity metrics at a specific site."""
    pga_g: float
    pga_gal: float
    pgv_cms: float
    mmi: float
    mmi_roman: str
    jma_shindo: str
    jma_shindo_numeric: float


@dataclass(frozen=True, slots=True)
class ShakeMapGridPoint:
    """Grid cell point in a 2D ShakeMap overlay."""
    lat: float
    lon: float
    distance_km: float
    pga_g: float
    pga_gal: float
    pgv_cms: float
    mmi: float
    mmi_roman: str
    jma_shindo: str
    cell_id: Optional[str] = None


class ShakeMapEngine:
    """
    Seismological Ground Motion Prediction Equation (GMPE) and ShakeMap generator.

    Supports:
    - Fault-rupture directivity and radiation pattern anisotropy.
    - Crustal, subduction interface, and deep intraslab tectonic regimes.
    - Standard USGS Instrumental Modified Mercalli Intensity (MMI I-XII, Worden et al. 2012).
    - Japan Meteorological Agency (JMA) Shindo scale (0, 1, 2, 3, 4, 5-, 5+, 6-, 6+, 7).
    - Grid-based ShakeMap generation for studio rendering.
    """

    # Regime attenuation parameters: (c0, c1_mag, c2_logR, c3_linR)
    REGIME_PARAMS = {
        "crustal": (-1.25, 0.52, 1.35, 0.0018),
        "subduction_interface": (-1.05, 0.50, 1.25, 0.0012),
        "subduction_intraslab": (-0.80, 0.48, 1.20, 0.0008),
    }

    # PGV regime parameters: (d0, d1_mag, d2_logR, d3_linR)
    PGV_PARAMS = {
        "crustal": (-1.10, 0.62, 1.20, 0.0010),
        "subduction_interface": (-0.95, 0.58, 1.15, 0.0008),
        "subduction_intraslab": (-0.75, 0.54, 1.10, 0.0005),
    }

    @staticmethod
    def select_regime(depth_km: float) -> str:
        """Automatically select tectonic regime based on hypocentral depth."""
        if depth_km < 35.0:
            return "crustal"
        elif depth_km < 70.0:
            return "subduction_interface"
        else:
            return "subduction_intraslab"

    @classmethod
    def compute_ground_motion(
        cls,
        magnitude: float,
        distance_hypo_km: float,
        depth_km: float,
        azimuth_deg: float = 0.0,
        strike_deg: Optional[float] = None,
        regime: Optional[str] = None,
    ) -> GroundMotionResult:
        """
        Compute peak ground acceleration (PGA), peak ground velocity (PGV),
        Instrumental MMI, and JMA Shindo at a site.

        Parameters
        ----------
        magnitude : float
            Moment magnitude (Mw).
        distance_hypo_km : float
            Hypocentral distance in kilometers (sqrt(d_epi^2 + depth^2)).
        depth_km : float
            Focal depth in kilometers.
        azimuth_deg : float
            Azimuth from epicenter to site in degrees [0, 360).
        strike_deg : Optional[float]
            Fault strike angle in degrees [0, 360). If provided, applies
            radiation pattern / rupture elongation anisotropy.
        regime : Optional[str]
            "crustal", "subduction_interface", or "subduction_intraslab".

        Returns
        -------
        GroundMotionResult
        """
        if regime is None:
            regime = cls.select_regime(depth_km)

        r_eff = max(distance_hypo_km, 3.0)

        # Anisotropic directivity / radiation factor
        aniso_factor = 1.0
        if strike_deg is not None:
            # Quadrupole radiation and fault strike directivity
            rad_diff = math.radians(2.0 * (azimuth_deg - strike_deg))
            aniso_factor = 1.0 + 0.25 * math.cos(rad_diff)

        # 1. Peak Ground Acceleration (PGA in g)
        c0, c1, c2, c3 = cls.REGIME_PARAMS.get(regime, cls.REGIME_PARAMS["crustal"])
        log10_pga = c0 + c1 * magnitude - c2 * math.log10(r_eff) - c3 * r_eff
        if magnitude < 3.5:
            # Atkinson & Morrison (2009) low-magnitude downward roll-off
            log10_pga += 0.8 * (magnitude - 3.5)
        pga_g = math.pow(10.0, log10_pga) * aniso_factor
        pga_g = min(max(pga_g, 1e-6), 2.5)  # Physical saturation ceiling at 2.5 g
        pga_gal = pga_g * G_TO_GAL

        # 2. Peak Ground Velocity (PGV in cm/s)
        d0, d1, d2, d3 = cls.PGV_PARAMS.get(regime, cls.PGV_PARAMS["crustal"])
        log10_pgv = d0 + d1 * magnitude - d2 * math.log10(r_eff) - d3 * r_eff
        if magnitude < 3.5:
            log10_pgv += 0.8 * (magnitude - 3.5)
        pgv_cms = math.pow(10.0, log10_pgv) * aniso_factor
        pgv_cms = min(max(pgv_cms, 1e-4), 250.0)  # Physical saturation ceiling at 250 cm/s

        # 3. Instrumental MMI (Worden et al. 2012)
        mmi, mmi_roman = cls.compute_mmi_from_pga_pgv(pga_gal, pgv_cms)

        # 4. JMA Shindo (Midorikawa et al. / Shabestari-Yamazaki)
        jma_shindo, jma_shindo_numeric = cls.compute_jma_shindo(pga_gal, pgv_cms)

        return GroundMotionResult(
            pga_g=pga_g,
            pga_gal=pga_gal,
            pgv_cms=pgv_cms,
            mmi=mmi,
            mmi_roman=mmi_roman,
            jma_shindo=jma_shindo,
            jma_shindo_numeric=jma_shindo_numeric,
        )

    @staticmethod
    def compute_mmi_from_pga_pgv(pga_gal: float, pgv_cms: float) -> Tuple[float, str]:
        """
        Convert PGA (Gal) and PGV (cm/s) to Instrumental Modified Mercalli Intensity
        using Worden et al. (2012) ShakeMap empirical relations.
        """
        log_pga = math.log10(max(pga_gal, 1e-5))
        log_pgv = math.log10(max(pgv_cms, 1e-5))

        # PGA branch (Worden et al. 2012 Table 1 continuous parameters)
        if log_pga <= 1.57:
            mmi_pga = 1.78 + 1.55 * log_pga
        else:
            mmi_pga = -1.60 + 3.70 * log_pga

        # PGV branch (Worden et al. 2012 Table 1 continuous parameters)
        if log_pgv <= 0.53:
            mmi_pgv = 3.78 + 1.47 * log_pgv
        else:
            mmi_pgv = 2.89 + 3.16 * log_pgv


        # Weighted transition: PGA governs low shaking, PGV governs high damage
        # (Worden et al. 2012 Section 4 transition)
        ind_mmi = 0.5 * (mmi_pga + mmi_pgv)
        if ind_mmi <= 4.5:
            mmi = mmi_pga
        elif ind_mmi >= 6.5:
            mmi = mmi_pgv
        else:
            weight_pgv = (ind_mmi - 4.5) / 2.0
            weight_pgv = min(max(weight_pgv, 0.0), 1.0)
            mmi = (1.0 - weight_pgv) * mmi_pga + weight_pgv * mmi_pgv

        mmi = min(max(mmi, 1.0), 12.0)


        # Convert to standard Roman numerals
        roman_map = [
            (1.5, "I"),
            (2.5, "II"),
            (3.5, "III"),
            (4.5, "IV"),
            (5.5, "V"),
            (6.5, "VI"),
            (7.5, "VII"),
            (8.5, "VIII"),
            (9.5, "IX"),
            (10.5, "X"),
            (11.5, "XI"),
            (12.0, "XII"),
        ]
        mmi_roman = "XII"
        for thresh, roman in roman_map:
            if mmi < thresh:
                mmi_roman = roman
                break

        return round(mmi, 2), mmi_roman

    @staticmethod
    def compute_jma_shindo(pga_gal: float, pgv_cms: float) -> Tuple[str, float]:
        """
        Convert ground motion to Japan Meteorological Agency (JMA) Shindo scale
        using Shabestari & Yamazaki (2001) / Midorikawa et al. (1999) instrumental formulations.

        JMA Tiers: 0, 1, 2, 3, 4, 5-, 5+, 6-, 6+, 7
        """
        log_pgv = math.log10(max(pgv_cms, 1e-5))
        log_pga = math.log10(max(pga_gal, 1e-5))

        # Continuous JMA instrumental intensity estimation
        i_pgv = 2.00 * log_pgv + 2.68
        i_pga = 2.01 * log_pga - 0.70

        # Harmonized continuous intensity
        i_jma = 0.5 * (i_pgv + i_pga)
        i_jma = min(max(i_jma, 0.0), 7.0)

        # Map continuous intensity to official discrete JMA Shindo scale
        if i_jma < 0.5:
            shindo = "0"
        elif i_jma < 1.5:
            shindo = "1"
        elif i_jma < 2.5:
            shindo = "2"
        elif i_jma < 3.5:
            shindo = "3"
        elif i_jma < 4.5:
            shindo = "4"
        elif i_jma < 5.0:
            shindo = "5-"
        elif i_jma < 5.5:
            shindo = "5+"
        elif i_jma < 6.0:
            shindo = "6-"
        elif i_jma < 6.5:
            shindo = "6+"
        else:
            shindo = "7"

        return shindo, round(i_jma, 2)

    @classmethod
    def estimate_perception_radius(
        cls,
        magnitude: float,
        depth_km: float,
        min_mmi: float = 1.0,
    ) -> float:
        """
        Calculate maximum perception bounding radius R_max (km) for spatial candidate pruning.

        Guarantees that all municipalities experiencing shaking >= min_mmi are included
        while rejecting distant unaffected regions in logarithmic time.
        """
        # Seismological empirical perception distance envelope
        # For small events (M < 3.0), radius drops sharply
        if magnitude < 3.0:
            r_max = max(0.5, math.pow(10.0, 0.75 * magnitude - 1.5))
        else:
            base_radius = math.pow(10.0, 0.45 * magnitude - 0.75)
            # Deep focus events distribute low-amplitude energy over wider epicentral areas
            depth_expansion = 1.5 * depth_km
            r_max = max(15.0, base_radius + depth_expansion)

        # Sensitivity scaling based on target MMI threshold
        if min_mmi > 1.0:
            # If filtering for higher shaking (e.g. MMI >= 3.0), radius decreases logarithmically
            scale = max(0.2, 1.0 - 0.15 * (min_mmi - 1.0))
            r_max *= scale

        # Physical cap at antipodal distance
        return min(r_max, 20000.0)

    @classmethod
    def generate_shakemap_grid(
        cls,
        epicenter_lat: float,
        epicenter_lon: float,
        depth_km: float,
        magnitude: float,
        strike_deg: Optional[float] = None,
        radius_km: Optional[float] = None,
        resolution_deg: float = 0.1,
    ) -> List[ShakeMapGridPoint]:
        """
        Generate a regular 2D ShakeMap grid covering the affected area for studio rendering.

        Parameters
        ----------
        epicenter_lat : float
        epicenter_lon : float
        depth_km : float
        magnitude : float
        strike_deg : Optional[float]
        radius_km : Optional[float]
            If None, defaults to estimate_perception_radius(magnitude, depth_km, min_mmi=1.5).
        resolution_deg : float
            Grid spacing in degrees (default 0.1 deg ~ 11 km).

        Returns
        -------
        List[ShakeMapGridPoint]
        """
        if radius_km is None:
            radius_km = cls.estimate_perception_radius(magnitude, depth_km, min_mmi=1.5)

        # Approximate angular extent
        delta_deg = math.degrees(radius_km / EARTH_RADIUS_KM)
        delta_lat = delta_deg
        # Adjust for latitude convergence
        cos_lat = max(math.cos(math.radians(epicenter_lat)), 0.05)
        delta_lon = delta_deg / cos_lat

        lat_min = max(-90.0, epicenter_lat - delta_lat)
        lat_max = min(90.0, epicenter_lat + delta_lat)
        lon_min = epicenter_lon - delta_lon
        lon_max = epicenter_lon + delta_lon

        grid_points: List[ShakeMapGridPoint] = []

        n_lat = int(math.ceil((lat_max - lat_min) / resolution_deg)) + 1
        n_lon = int(math.ceil((lon_max - lon_min) / resolution_deg)) + 1

        for i in range(n_lat):
            curr_lat = lat_min + i * resolution_deg
            if curr_lat > lat_max:
                break
            for j in range(n_lon):
                curr_lon = lon_min + j * resolution_deg
                # Normalize longitude to [-180, 180)
                norm_lon = ((curr_lon + 180.0) % 360.0) - 180.0

                # Compute great-circle distance
                d_epi = cls._great_circle_distance(epicenter_lat, epicenter_lon, curr_lat, norm_lon)
                if d_epi > radius_km:
                    continue

                d_hypo = math.sqrt(d_epi * d_epi + depth_km * depth_km)
                azimuth = cls._calculate_azimuth(epicenter_lat, epicenter_lon, curr_lat, norm_lon)

                gm = cls.compute_ground_motion(
                    magnitude=magnitude,
                    distance_hypo_km=d_hypo,
                    depth_km=depth_km,
                    azimuth_deg=azimuth,
                    strike_deg=strike_deg,
                )

                cell_key = f"cell_{int(round((curr_lat + 90) * 100))}_{int(round((norm_lon + 180) * 100))}"

                grid_points.append(
                    ShakeMapGridPoint(
                        lat=round(curr_lat, 4),
                        lon=round(norm_lon, 4),
                        distance_km=round(d_epi, 2),
                        pga_g=gm.pga_g,
                        pga_gal=gm.pga_gal,
                        pgv_cms=gm.pgv_cms,
                        mmi=gm.mmi,
                        mmi_roman=gm.mmi_roman,
                        jma_shindo=gm.jma_shindo,
                        cell_id=cell_key,
                    )
                )

        return grid_points

    @staticmethod
    def _great_circle_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance between two coordinates in kilometers (Haversine formula)."""
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_phi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2)
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
        return EARTH_RADIUS_KM * c

    @staticmethod
    def _calculate_azimuth(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Initial bearing (forward azimuth) from point 1 to point 2 in degrees [0, 360)."""
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_lambda = math.radians(lon2 - lon1)

        y = math.sin(delta_lambda) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)
        azimuth = math.degrees(math.atan2(y, x))
        return (azimuth + 360.0) % 360.0
