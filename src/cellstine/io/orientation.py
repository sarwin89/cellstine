"""Helpers for canonicalizing slab-like structures into the xy plane."""

from __future__ import annotations

import numpy as np

from ..core.transforms import right_handed_lattice
from . import native as native_vasp
from .models import StructureRecord


class OrientationNormalizer:
    """Rotate structures so the c axis points along +z when requested."""

    def ensure_right_handed(self, record: StructureRecord) -> StructureRecord:
        lattice, positions_direct = right_handed_lattice(record.lattice, record.positions_direct)
        positions_cartesian = native_vasp.direct_to_cartesian(positions_direct, lattice)
        updated = record.copy()
        updated.lattice = lattice
        updated.positions_direct = positions_direct
        updated.positions_cartesian = positions_cartesian
        updated.metadata["right_handed"] = True
        return updated

    def align_ab_to_xy(self, record: StructureRecord) -> StructureRecord:
        """Rotate a structure so ``a`` lies along ``+x`` and ``b`` lies in the ``xy`` plane.

        This is the convention every slab stage needs: the surface plane is the
        ``xy`` plane, heights are Cartesian ``z``, and the two in-plane lattice
        vectors form the 2x2 basis the matching and building code works with.
        Aligning ``c`` to ``z`` instead only coincides with it for a cell whose
        ``c`` is already perpendicular to the surface, and leaves ``a`` and ``b``
        out of the plane for any tilted cell.  The operation is a rigid rotation,
        so every interatomic distance is unchanged.

        The frame is proved to be a rotation, and the three components zeroed
        below are proved to be exactly zero, in
        ``aristotle-lean-reference/RequestProject/SurfaceAlignment.lean``
        (``Cellstine.alignFrame_det_eq_one``,
        ``Cellstine.alignFrame_preserves_dist_sq``,
        ``Cellstine.alignFrame_apply_first``,
        ``Cellstine.alignFrame_apply_second_height``,
        ``Cellstine.alignFrame_apply_second_pos``,
        ``Cellstine.alignFrame_height``); the zeroing is a clean-up of
        floating-point noise, not a correction.
        """

        updated = self.ensure_right_handed(record)
        lattice = np.asarray(updated.lattice, dtype=float)
        cartesian = np.asarray(updated.positions_cartesian, dtype=float)
        normal = np.cross(lattice[0], lattice[1])
        normal_length = float(np.linalg.norm(normal))
        first_length = float(np.linalg.norm(lattice[0]))
        if normal_length <= 1e-12 or first_length <= 1e-12:
            raise ValueError("the a and b lattice vectors must span a plane")
        z_hat = normal / normal_length
        x_hat = lattice[0] / first_length
        y_hat = np.cross(z_hat, x_hat)
        rotation = np.column_stack((x_hat, y_hat, z_hat))
        rotated_lattice = lattice @ rotation
        rotated_lattice[0, 1:] = 0.0
        rotated_lattice[1, 2] = 0.0
        rotated_cartesian = cartesian @ rotation
        updated.lattice = rotated_lattice
        updated.positions_cartesian = rotated_cartesian
        updated.positions_direct = native_vasp.cartesian_to_direct(rotated_cartesian, rotated_lattice)
        updated.coordinate_mode = "Direct"
        updated.metadata["aligned_ab_to_xy"] = True
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
        rotated_direct = native_vasp.cartesian_to_direct(rotated_cartesian, rotated_lattice)
        updated.lattice = rotated_lattice
        updated.positions_cartesian = rotated_cartesian
        updated.positions_direct = rotated_direct
        updated.coordinate_mode = "Direct"
        updated.metadata["aligned_c_to_z"] = True
        return self.ensure_right_handed(updated)
