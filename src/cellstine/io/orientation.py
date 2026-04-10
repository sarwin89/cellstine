"""Helpers for canonicalizing slab-like structures into the xy plane."""

from __future__ import annotations

import numpy as np

from ..core.base import legacy_modules
from ..core.transforms import right_handed_lattice
from .models import StructureRecord


class OrientationNormalizer:
    """Rotate structures so the c axis points along +z when requested."""

    def ensure_right_handed(self, record: StructureRecord) -> StructureRecord:
        lattice, positions_direct = right_handed_lattice(record.lattice, record.positions_direct)
        positions_cartesian = legacy_modules().io_mod.direct_to_cartesian(positions_direct, lattice)
        updated = record.copy()
        updated.lattice = lattice
        updated.positions_direct = positions_direct
        updated.positions_cartesian = positions_cartesian
        updated.metadata["right_handed"] = True
        return updated

    def align_c_to_z(self, record: StructureRecord) -> StructureRecord:
        updated = self.ensure_right_handed(record)
        lattice = np.asarray(updated.lattice, dtype=float)
        cartesian = np.asarray(updated.positions_cartesian, dtype=float)
        c_vec = lattice[2]
        c_norm = float(np.linalg.norm(c_vec))
        if c_norm <= 1e-15:
            return updated
        z_hat = c_vec / c_norm
        x_trial = lattice[0] - np.dot(lattice[0], z_hat) * z_hat
        if float(np.linalg.norm(x_trial)) <= 1e-12:
            x_trial = lattice[1] - np.dot(lattice[1], z_hat) * z_hat
        x_hat = x_trial / max(float(np.linalg.norm(x_trial)), 1e-12)
        y_hat = np.cross(z_hat, x_hat)
        y_hat /= max(float(np.linalg.norm(y_hat)), 1e-12)
        rotation = np.column_stack((x_hat, y_hat, z_hat))
        rotated_lattice = lattice @ rotation
        rotated_cartesian = cartesian @ rotation
        rotated_direct = legacy_modules().io_mod.cartesian_to_direct(rotated_cartesian, rotated_lattice)
        updated.lattice = rotated_lattice
        updated.positions_cartesian = rotated_cartesian
        updated.positions_direct = rotated_direct
        updated.coordinate_mode = "Direct"
        updated.metadata["aligned_c_to_z"] = True
        return self.ensure_right_handed(updated)
