"""SO(6) Kelvin-Mandel Stiffness Tensor Rotors across Tilted Subducting Slabs.

Grounded in:
- Kelvin-Mandel normalized Voigt basis:
  sigma_KM = (sigma_11, sigma_22, sigma_33, sqrt(2)*sigma_23, sqrt(2)*sigma_13, sqrt(2)*sigma_12)^T.
- Lift of SO(3) bedding rotation R into SO(6) orthogonal rotor M(R):
  M(R)^T M(R) = I_6, det(M(R)) = +1, and Tr(M C M^T) = Tr(C).
- Exact preservation of elastic strain energy: 1/2 eps^T C eps = 1/2 eps'^T C' eps'.
- Thomsen (1986) TTI phase velocity along ray tangent direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from quakesnail._quakesnail_native import py_so6_kelvin_mandel_rotor


def compute_so3_rotation(dip_deg: float, azimuth_deg: float) -> np.ndarray:
    """Compute 3x3 rotation matrix in SO(3) from slab dip and azimuth angles."""
    dip = np.radians(dip_deg)
    az = np.radians(azimuth_deg)

    # Rotation around Z by -azimuth, then around Y by dip
    rz = np.array([
        [np.cos(az), -np.sin(az), 0.0],
        [np.sin(az),  np.cos(az), 0.0],
        [0.0,         0.0,        1.0],
    ], dtype=np.float32)

    ry = np.array([
        [ np.cos(dip), 0.0, np.sin(dip)],
        [ 0.0,         1.0, 0.0        ],
        [-np.sin(dip), 0.0, np.cos(dip)],
    ], dtype=np.float32)

    return np.dot(rz, ry).astype(np.float32)


def compute_so6_rotor(dip_deg: float, azimuth_deg: float) -> np.ndarray:
    """Compute 6x6 orthogonal Kelvin-Mandel rotor M(R) in SO(6)."""
    r_so3 = compute_so3_rotation(dip_deg, azimuth_deg)
    return py_so6_kelvin_mandel_rotor(r_so3)


@dataclass
class StiffnessRotor:
    """Manages SO(6) Kelvin-Mandel stiffness tensors and anisotropic TTI phase velocities."""

    dip_deg: float = 45.0
    azimuth_deg: float = 90.0
    vp0_kms: float = 8.0
    vs0_kms: float = 4.6
    density_gcc: float = 3.3
    epsilon_thomsen: float = 0.08  # P-wave anisotropy
    delta_thomsen: float = 0.03    # P-wave onset curvature
    gamma_thomsen: float = 0.05    # S-wave anisotropy

    def __post_init__(self):
        self.rotor_so6 = compute_so6_rotor(self.dip_deg, self.azimuth_deg)

    def phase_velocity_p(self, ray_tangent: np.ndarray) -> float:
        """Compute directional P-wave phase velocity along 3D ray unit tangent vector.

        V_p(psi) = V_p0 * (1 + delta * sin^2(psi) * cos^2(psi) + epsilon * sin^4(psi))
        where psi is angle between ray tangent and the tilted slab normal axis.
        """
        t = np.asarray(ray_tangent, dtype=np.float32)
        norm_t = np.linalg.norm(t)
        if norm_t < 1e-6:
            return float(self.vp0_kms)
        hat_t = t / norm_t

        # Slab symmetry axis (normal to bedding)
        dip = np.radians(self.dip_deg)
        az = np.radians(self.azimuth_deg)
        n_axis = np.array([
            np.sin(dip) * np.cos(az),
            np.sin(dip) * np.sin(az),
            np.cos(dip),
        ], dtype=np.float32)

        cos_psi = float(np.clip(abs(np.dot(hat_t, n_axis)), 0.0, 1.0))
        sin_psi_sq = max(0.0, 1.0 - cos_psi**2)

        # Thomsen weak anisotropy formulation
        vp = self.vp0_kms * (
            1.0
            + self.delta_thomsen * sin_psi_sq * (cos_psi**2)
            + self.epsilon_thomsen * (sin_psi_sq**2)
        )
        return float(vp)

    def phase_velocity_s(self, ray_tangent: np.ndarray) -> float:
        """Compute directional S-wave phase velocity along 3D ray unit tangent vector."""
        t = np.asarray(ray_tangent, dtype=np.float32)
        norm_t = np.linalg.norm(t)
        if norm_t < 1e-6:
            return float(self.vs0_kms)
        hat_t = t / norm_t

        dip = np.radians(self.dip_deg)
        az = np.radians(self.azimuth_deg)
        n_axis = np.array([
            np.sin(dip) * np.cos(az),
            np.sin(dip) * np.sin(az),
            np.cos(dip),
        ], dtype=np.float32)

        cos_psi = float(np.clip(abs(np.dot(hat_t, n_axis)), 0.0, 1.0))
        sin_psi_sq = max(0.0, 1.0 - cos_psi**2)

        vs = self.vs0_kms * (1.0 + self.gamma_thomsen * sin_psi_sq)
        return float(vs)

    def rotate_stiffness_matrix(self, c_vti: np.ndarray) -> np.ndarray:
        """Rotate 6x6 VTI stiffness tensor into 3D TTI tensor: C_TTI = M(R) * C_VTI * M(R)^T."""
        c = np.asarray(c_vti, dtype=np.float32)
        m = self.rotor_so6
        return np.dot(m, np.dot(c, m.T)).astype(np.float32)
