"""3D Global Spherical Raytracing across 1D PREM + 3D Subducting Slab + Mantle Plume.

Grounded in:
- Dziewonski & Anderson (1981) Preliminary Reference Earth Model (PREM).
- Um & Thurber (1987) 3-Point Minimum-Time Pseudo-Bending Raytracing.
- 3D Subducting Slab (+3.5% tabular body) & Thermal Mantle Plume (-3.0% conduit).
- Direct P, direct S, Moho reflection (PmP), CMB reflection (PcP), and core phases (PKP, PKIKP).
- Cross-spoke compatibility with `g3vm.raytrace_let`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from quakesnail._quakesnail_native import (
    EARTH_RADIUS_KM,
    DEG_TO_RAD,
    PHASE_P,
    PHASE_S,
    PHASE_PKP,
    PHASE_PKIKP,
    PHASE_PMP,
    PHASE_PCP,
    py_prem_vp_kms,
)


@dataclass
class RayPath:
    phase_name: str
    travel_time_s: float
    nodes_xyz_km: np.ndarray  # (N, 3) Cartesian coordinates
    nodes_lat_lon_depth: np.ndarray  # (N, 3) [lat_deg, lon_deg, depth_km]
    length_km: float = 0.0
    turning_depth_km: float = 0.0


def prem_vp_kms(r_km: float) -> float:
    """1D PREM compressional P-wave velocity."""
    return py_prem_vp_kms(r_km)


def prem_vs_kms(r_km: float) -> float:
    """1D PREM shear S-wave velocity."""
    r = float(max(0.0, min(EARTH_RADIUS_KM, r_km)))
    x = r / EARTH_RADIUS_KM

    if r < 1221.5:
        return float(3.6678 - 4.4475 * (x**2))
    elif r < 3480.0:
        return 0.0001
    elif r < 5701.0:
        return float(8.2435 - 3.4922 * x + 3.0805 * (x**2) - 1.4886 * (x**3))
    elif r < 5771.0:
        return float(13.1118 - 21.0827 * x + 12.3965 * (x**2))
    elif r < 5971.0:
        return float(15.0298 - 11.7580 * x)
    elif r < 6151.0:
        return float(16.1477 - 13.0645 * x)
    elif r < 6346.6:
        return float(13.5670 - 9.1720 * x)
    elif r < 6356.0:
        return 3.9000
    else:
        return 3.2000


def lat_lon_depth_to_xyz(lat_deg: float, lon_deg: float, depth_km: float) -> np.ndarray:
    """Convert spherical (lat, lon, depth) to Cartesian (x, y, z) in km."""
    r = float(max(10.0, EARTH_RADIUS_KM - depth_km))
    phi = np.radians(lat_deg)
    lam = np.radians(lon_deg)
    x = r * np.cos(phi) * np.cos(lam)
    y = r * np.cos(phi) * np.sin(lam)
    z = r * np.sin(phi)
    return np.array([x, y, z], dtype=np.float32)


def xyz_to_lat_lon_depth(xyz: np.ndarray) -> np.ndarray:
    """Convert Cartesian (x, y, z) in km to (lat, lon, depth)."""
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    r = float(np.sqrt(x**2 + y**2 + z**2))
    lat = float(np.degrees(np.arcsin(np.clip(z / max(1.0, r), -1.0, 1.0))))
    lon = float(np.degrees(np.arctan2(y, x)))
    depth = float(EARTH_RADIUS_KM - r)
    return np.array([lat, lon, depth], dtype=np.float32)


class SphericalRaytracer:
    """3D Minimum-Time Spherical Raytracer across PREM + Subducting Slab + Mantle Plume."""

    def __init__(
        self,
        slab_dip_deg: float = 45.0,
        slab_azimuth_deg: float = 90.0,
        enable_slab: bool = True,
        enable_plume: bool = True,
    ):
        self.slab_dip_deg = float(slab_dip_deg)
        self.slab_azimuth_deg = float(slab_azimuth_deg)
        self.enable_slab = enable_slab
        self.enable_plume = enable_plume

    def velocity_3d(self, xyz_km: np.ndarray, is_s_wave: bool = False) -> float:
        """Evaluate 3D isotropic velocity at Cartesian location xyz."""
        r = float(np.linalg.norm(xyz_km))
        base_v = prem_vs_kms(r) if is_s_wave else prem_vp_kms(r)

        delta = 0.0

        # 1. Subducting Slab: +3.5% fast tabular body
        if self.enable_slab and r > 3480.0:
            dip_rad = np.radians(self.slab_dip_deg)
            az_rad = np.radians(self.slab_azimuth_deg)
            n_slab = np.array([
                np.sin(dip_rad) * np.cos(az_rad),
                np.sin(dip_rad) * np.sin(az_rad),
                np.cos(dip_rad),
            ], dtype=np.float32)

            slab_dist = abs(float(np.dot(xyz_km - np.array([0, 0, EARTH_RADIUS_KM]), n_slab)))
            sigma_slab = 45.0  # 45 km half-thickness
            delta += 0.035 * np.exp(-0.5 * (slab_dist / sigma_slab)**2)

        # 2. Thermal Mantle Plume: -3.0% slow cylindrical anomaly
        if self.enable_plume and 3480.0 < r < 6000.0:
            d_axis = float(np.sqrt(xyz_km[0]**2 + xyz_km[1]**2))
            r_plume = 100.0  # 100 km plume radius
            delta -= 0.030 * np.exp(-0.5 * (d_axis / r_plume)**2)

        return float(base_v * (1.0 + delta))

    def trace(
        self,
        src_lat_lon_depth: Tuple[float, float, float],
        rcv_lat_lon_depth: Tuple[float, float, float],
        phase: str = "P",
        num_nodes: int = 32,
        iterations: int = 8,
    ) -> RayPath:
        """Trace minimum-time 3D ray path between source and receiver."""
        phase_upper = phase.upper()
        is_s = "S" in phase_upper and "PK" not in phase_upper

        p_src = lat_lon_depth_to_xyz(*src_lat_lon_depth)
        p_rcv = lat_lon_depth_to_xyz(*rcv_lat_lon_depth)

        n = max(8, min(64, num_nodes))
        nodes = np.zeros((n, 3), dtype=np.float32)

        # Handle Moho reflection PmP or CMB reflection PcP
        if phase_upper == "PMP":
            r_moho = EARTH_RADIUS_KM - 35.0
            midpoint = 0.5 * (p_src + p_rcv)
            midpoint = midpoint / np.linalg.norm(midpoint) * r_moho
            for i in range(n // 2):
                frac = i / (n // 2)
                nodes[i] = p_src + frac * (midpoint - p_src)
            for i in range(n // 2, n):
                frac = (i - n // 2) / (n - n // 2 - 1)
                nodes[i] = midpoint + frac * (p_rcv - midpoint)
        elif phase_upper == "PCP":
            r_cmb = 3480.0
            midpoint = 0.5 * (p_src + p_rcv)
            midpoint = midpoint / np.linalg.norm(midpoint) * r_cmb
            for i in range(n // 2):
                frac = i / (n // 2)
                nodes[i] = p_src + frac * (midpoint - p_src)
            for i in range(n // 2, n):
                frac = (i - n // 2) / (n - n // 2 - 1)
                nodes[i] = midpoint + frac * (p_rcv - midpoint)
        elif phase_upper in ("PKP", "PKIKP"):
            # Physical Core-Mantle Boundary (CMB) refraction path across fluid core and ICB
            r_cmb = 3480.0
            r_src = float(np.linalg.norm(p_src))
            r_rcv = float(np.linalg.norm(p_rcv))
            dot_val = float(np.clip(np.dot(p_src, p_rcv) / (r_src * r_rcv), -1.0, 1.0))
            delta_rad = float(np.arccos(dot_val))
            delta_deg = float(np.degrees(delta_rad))

            n_plane = np.cross(p_src, p_rcv)
            norm_n = float(np.linalg.norm(n_plane))
            if norm_n > 1e-6:
                n_plane = n_plane / norm_n
                u_src = p_src / r_src
                u_tan = np.cross(n_plane, u_src)
                u_tan = u_tan / np.linalg.norm(u_tan)
            else:
                u_src = p_src / r_src
                u_tan = np.array([0.0, 1.0, 0.0], dtype=np.float32)

            theta_m_deg = min(20.0, delta_deg * 0.14)
            theta_m_rad = np.radians(theta_m_deg)
            p_cmb_in = (np.cos(theta_m_rad) * u_src + np.sin(theta_m_rad) * u_tan) * r_cmb

            theta_exit_rad = delta_rad - theta_m_rad
            p_cmb_out = (np.cos(theta_exit_rad) * u_src + np.sin(theta_exit_rad) * u_tan) * r_cmb

            n1 = n // 4
            n3 = n // 4
            n2 = n - n1 - n3
            for i in range(n1):
                frac = i / n1
                nodes[i] = p_src + frac * (p_cmb_in - p_src)
            for i in range(n2):
                frac = i / (n2 - 1) if n2 > 1 else 0.0
                nodes[n1 + i] = p_cmb_in + frac * (p_cmb_out - p_cmb_in)
            for i in range(n3):
                frac = (i + 1) / n3
                nodes[n1 + n2 + i] = p_cmb_out + frac * (p_rcv - p_cmb_out)
        else:
            # Straight-line chord initialization
            for i in range(n):
                frac = i / (n - 1)
                nodes[i] = p_src + frac * (p_rcv - p_src)

        # Pseudo-bending iterations
        h = 2.0  # 2 km finite difference step
        for _it in range(iterations):
            for k in range(1, n - 1):
                p_prev = nodes[k - 1]
                p_next = nodes[k + 1]
                p_cur = nodes[k]

                chord = p_next - p_prev
                chord_l = float(np.linalg.norm(chord))
                if chord_l < 1e-3:
                    continue
                hat_t = chord / chord_l

                # Evaluate slowness and numerical gradient
                v_c = max(1.0, self.velocity_3d(p_cur, is_s))
                u_c = 1.0 / v_c

                grad_u = np.zeros(3, dtype=np.float32)
                for axis in range(3):
                    dp = np.zeros(3, dtype=np.float32)
                    dp[axis] = h
                    u_p = 1.0 / max(1.0, self.velocity_3d(p_cur + dp, is_s))
                    u_m = 1.0 / max(1.0, self.velocity_3d(p_cur - dp, is_s))
                    grad_u[axis] = (u_p - u_m) / (2.0 * h)

                norm_grad = grad_u / u_c
                perp = norm_grad - np.dot(norm_grad, hat_t) * hat_t

                # Um & Thurber midpoint displacement
                disp = -(chord_l**2 / 8.0) * perp
                disp = np.clip(disp, -15.0, 15.0)

                mid = 0.5 * (p_prev + p_next)
                nodes[k] = mid + disp

        # Compute travel time and segment lengths
        total_time = 0.0
        total_length = 0.0
        max_depth = 0.0
        ll_depths = np.zeros((n, 3), dtype=np.float32)

        for i in range(n):
            ll_depths[i] = xyz_to_lat_lon_depth(nodes[i])
            max_depth = max(max_depth, float(ll_depths[i, 2]))

        for i in range(n - 1):
            seg_l = float(np.linalg.norm(nodes[i + 1] - nodes[i]))
            total_length += seg_l
            mid_p = 0.5 * (nodes[i] + nodes[i + 1])
            v_mid = max(1.0, self.velocity_3d(mid_p, is_s))
            total_time += seg_l / v_mid

        return RayPath(
            phase_name=phase_upper,
            travel_time_s=float(total_time),
            nodes_xyz_km=nodes,
            nodes_lat_lon_depth=ll_depths,
            length_km=float(total_length),
            turning_depth_km=float(max_depth),
        )


def trace_bent_ray(
    source_xyz: Tuple[float, float, float],
    receiver_xyz: Tuple[float, float, float],
    phase_name: str = "P",
    velocity_model: Optional[Any] = None,
) -> RayPath:
    """Cross-spoke bridge matching PROJECT.md § Interface Contracts (g3vm.raytrace_let)."""
    tracer = SphericalRaytracer()
    src_ll = xyz_to_lat_lon_depth(np.array(source_xyz, dtype=np.float32))
    rcv_ll = xyz_to_lat_lon_depth(np.array(receiver_xyz, dtype=np.float32))
    return tracer.trace(
        (float(src_ll[0]), float(src_ll[1]), float(src_ll[2])),
        (float(rcv_ll[0]), float(rcv_ll[1]), float(rcv_ll[2])),
        phase=phase_name,
    )


def velocity_3d_p(x_km: float, y_km: float, z_km: float, slab_dip_deg: float = 45.0, slab_az_deg: float = 90.0) -> float:
    """3D Velocity Field with Subducting Slab (+3.5%) and Mantle Plume (-3.0%)."""
    tracer = SphericalRaytracer(slab_dip_deg=slab_dip_deg, slab_azimuth_deg=slab_az_deg, enable_slab=True, enable_plume=True)
    return tracer.velocity_3d(np.array([x_km, y_km, z_km], dtype=np.float32), is_s_wave=False)


def pseudo_bending_raytrace(
    src_xyz: Tuple[float, float, float] | np.ndarray,
    rcv_xyz: Tuple[float, float, float] | np.ndarray,
    n_nodes: int = 32,
    n_iter: int = 6,
) -> Tuple[np.ndarray, float]:
    """Um & Thurber (1987) 3-Point Pseudo-Bending Raytracer."""
    tracer = SphericalRaytracer(slab_dip_deg=45.0, slab_azimuth_deg=90.0, enable_slab=True, enable_plume=True)
    p_src = xyz_to_lat_lon_depth(np.asarray(src_xyz, dtype=np.float32))
    p_rcv = xyz_to_lat_lon_depth(np.asarray(rcv_xyz, dtype=np.float32))
    ray = tracer.trace(
        (float(p_src[0]), float(p_src[1]), float(p_src[2])),
        (float(p_rcv[0]), float(p_rcv[1]), float(p_rcv[2])),
        phase="P",
        num_nodes=n_nodes,
        iterations=n_iter,
    )
    return ray.nodes_xyz_km, float(ray.travel_time_s)
