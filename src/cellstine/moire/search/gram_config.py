"""Inputs, outputs and tolerances of the Gram-form moire search.

The search is configured by :class:`SearchConfig` and reports
:class:`SearchResult`; both are validated here, together with the shared
numerical tolerances that the stages of `cellstine.moire.search.gram` use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...core.symmetry2d import lattice_point_group, proper_subgroup

_TWO_PI = 2.0 * np.pi
_REL = 1e-9
_SLACK = 1e-9
_SHELL_RATIO = 1.6
_JOIN_CHUNK = 2048
_CERTIFICATION_MARGIN = 1e-10
# Width of the acceptance window when both strain budgets are zero.  A rigid
# search asks for Gram forms that are *equal*; the window is only there to
# absorb the rounding of the Cartesian bases, and it is wide enough for the
# certification margin above to still leave a certified band.
_RIGID_BAND = 1e-9


class SymmetricBranchUnavailable(ValueError):
    """Raised when the restricted square/hexagonal search cannot be used."""


def _validated_basis(value: np.ndarray, name: str) -> np.ndarray:
    basis = np.asarray(value, dtype=float)
    if basis.shape != (2, 2):
        raise ValueError(f"{name} must be a 2x2 Cartesian column basis")
    if not np.all(np.isfinite(basis)):
        raise ValueError(f"{name} must contain only finite values")
    magnitude = float(np.max(np.abs(basis)))
    if magnitude == 0.0:
        raise ValueError(f"{name} must be nonsingular")
    normalized = basis / magnitude
    determinant = float(np.linalg.det(normalized))
    column_scale = float(
        np.linalg.norm(normalized[:, 0]) * np.linalg.norm(normalized[:, 1])
    )
    if abs(determinant) <= 64.0 * np.finfo(float).eps * column_scale:
        raise ValueError(f"{name} must be nonsingular")
    validated = np.array(basis, copy=True)
    validated.setflags(write=False)
    return validated


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _validated_group(value: Any, basis: np.ndarray, name: str) -> np.ndarray:
    """Return a checked integer point group, defaulting to the bare lattice group.

    Every element must be an isometry of the given basis, that is
    ``G.T @ metric @ G == metric``.  Passing the *layer* group rather than the
    lattice group is what keeps three-fold layers such as hBN or MoS2 from being
    folded as if they were six-fold.
    """

    if value is None:
        group = lattice_point_group(basis)
    else:
        group = np.asarray(value, dtype=np.int64)
        if group.ndim == 2:
            group = group[None, :, :]
        if group.ndim != 3 or group.shape[1:] != (2, 2):
            raise ValueError(f"{name} must be an (k, 2, 2) integer array")
        metric = basis.T @ basis
        scale = float(np.max(np.abs(metric)))
        for element in group:
            determinant = int(element[0, 0] * element[1, 1] - element[0, 1] * element[1, 0])
            if abs(determinant) != 1:
                raise ValueError(f"{name} elements must be unimodular")
            transformed = element.T.astype(float) @ metric @ element.astype(float)
            if not np.allclose(transformed, metric, atol=1e-8 * max(scale, 1.0)):
                raise ValueError(f"{name} elements must preserve the lattice metric")
    identity = np.eye(2, dtype=np.int64)
    if not any(np.array_equal(element, identity) for element in group):
        group = np.concatenate([identity[None, :, :], group], axis=0)
    group = np.array(group, dtype=np.int64, copy=True)
    group.setflags(write=False)
    return group


@dataclass(frozen=True)
class SearchConfig:
    """Physical limits for a two-layer Gram-form search.

    Bases are finite nonsingular 2x2 Cartesian column bases.  Strain values are
    nonnegative principal logarithmic-strain budgets.  Both may be zero: that is
    a *rigid* search, which accepts only pairs whose Gram forms agree to within
    rounding, as a twisted homobilayer does at a commensurate angle.
    """

    top_basis: np.ndarray
    bottom_basis: np.ndarray
    max_length: float
    top_strain: float
    bottom_strain: float
    min_length: float | None = None
    max_atoms: int | None = None
    top_atoms: int = 1
    bottom_atoms: int = 1
    max_aspect_ratio: float = 12.0
    min_cell_angle_deg: float = 25.0
    max_cell_angle_deg: float = 155.0
    min_twist_angle_deg: float | None = None
    max_twist_angle_deg: float | None = None
    fold_symmetry: bool = True
    symmetric: bool = False
    primitive_only: bool = True
    top_group: np.ndarray | None = None
    bottom_group: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "top_basis", _validated_basis(self.top_basis, "top_basis"))
        object.__setattr__(
            self, "bottom_basis", _validated_basis(self.bottom_basis, "bottom_basis")
        )
        object.__setattr__(
            self, "top_group", _validated_group(self.top_group, self.top_basis, "top_group")
        )
        object.__setattr__(
            self,
            "bottom_group",
            _validated_group(self.bottom_group, self.bottom_basis, "bottom_group"),
        )
        numeric = {
            "max_length": self.max_length,
            "top_strain": self.top_strain,
            "bottom_strain": self.bottom_strain,
            "max_aspect_ratio": self.max_aspect_ratio,
            "min_cell_angle_deg": self.min_cell_angle_deg,
            "max_cell_angle_deg": self.max_cell_angle_deg,
        }
        if any(not np.isfinite(float(value)) for value in numeric.values()):
            raise ValueError("search limits and strain budgets must be finite")
        if self.max_length <= 0.0:
            raise ValueError("max_length must be positive")
        if self.top_strain < 0.0 or self.bottom_strain < 0.0:
            raise ValueError("strain budgets must be nonnegative")
        if self.min_length is not None:
            if not np.isfinite(float(self.min_length)) or self.min_length <= 0.0:
                raise ValueError("min_length must be finite and positive")
            if self.min_length > self.max_length:
                raise ValueError("min_length cannot exceed max_length")
        if self.max_atoms is not None:
            object.__setattr__(self, "max_atoms", _positive_integer(self.max_atoms, "max_atoms"))
        object.__setattr__(self, "top_atoms", _positive_integer(self.top_atoms, "top_atoms"))
        object.__setattr__(
            self, "bottom_atoms", _positive_integer(self.bottom_atoms, "bottom_atoms")
        )
        if self.max_aspect_ratio < 1.0:
            raise ValueError("max_aspect_ratio must be at least one")
        if not 0.0 < self.min_cell_angle_deg < self.max_cell_angle_deg < 180.0:
            raise ValueError("cell-angle limits must satisfy 0 < min < max < 180")
        for name in ("min_twist_angle_deg", "max_twist_angle_deg"):
            bound = getattr(self, name)
            if bound is None:
                continue
            value = float(bound)
            if not np.isfinite(value) or not 0.0 <= value <= 180.0:
                raise ValueError(f"{name} must be finite and lie in [0, 180]")
            object.__setattr__(self, name, value)
        if (
            self.min_twist_angle_deg is not None
            and self.max_twist_angle_deg is not None
            and self.min_twist_angle_deg > self.max_twist_angle_deg
        ):
            raise ValueError("min_twist_angle_deg cannot exceed max_twist_angle_deg")

    @property
    def _budget(self) -> float:
        return float(self.top_strain + self.bottom_strain)

    @property
    def is_rigid(self) -> bool:
        """True when no strain at all is allowed, so the search is exact."""

        return self._budget <= 0.0

    @property
    def _sharing(self) -> float:
        """Fraction of the relative strain carried by the top layer."""

        if self.is_rigid:
            return 0.5
        return float(self.top_strain) / self._budget

    @property
    def twist_window_radians(self) -> tuple[float, float] | None:
        """Window the *reported* twist magnitude must fall in, in radians.

        ``None`` means every commensurate twist the search finds is reported.
        The bounds are read on the folded angle, the one the candidate table
        shows, so a window is a statement about the bilayer rather than about
        the representative the enumeration happened to reach it by.
        """

        if self.min_twist_angle_deg is None and self.max_twist_angle_deg is None:
            return None
        lower = 0.0 if self.min_twist_angle_deg is None else float(self.min_twist_angle_deg)
        upper = 180.0 if self.max_twist_angle_deg is None else float(self.max_twist_angle_deg)
        return math.radians(lower), math.radians(upper)

    @property
    def angle_period_radians(self) -> float:
        """Twist-angle period implied by the two layer rotation symmetries."""

        top_order = len(proper_subgroup(self.top_group))
        bottom_order = len(proper_subgroup(self.bottom_group))
        return _TWO_PI / float(math.lcm(max(top_order, 1), max(bottom_order, 1)))

    @property
    def _band(self) -> tuple[float, float]:
        if self.is_rigid:
            return 1.0 - _RIGID_BAND, 1.0 + _RIGID_BAND
        lower = math.exp(-2.0 * self._budget)
        upper = math.exp(2.0 * self._budget)
        if lower <= 1.0 - _RIGID_BAND:
            return lower, upper
        return 1.0 - _RIGID_BAND, 1.0 + _RIGID_BAND


@dataclass(frozen=True)
class SearchResult:
    """Parallel arrays describing deterministic canonical candidate classes.

    ``principal_strains`` stores the two principal relative logarithmic strains.  The
    layer strains are obtained by multiplying by ``sharing_fraction`` for the top and by
    ``sharing_fraction - 1`` for the bottom.
    """

    top_matrices: np.ndarray
    bottom_matrices: np.ndarray
    top_gram: np.ndarray
    bottom_gram: np.ndarray
    twist_radians: np.ndarray
    twist_degrees: np.ndarray
    principal_strains: np.ndarray
    sharing_fraction: np.ndarray
    top_atom_counts: np.ndarray
    bottom_atom_counts: np.ndarray
    atom_counts: np.ndarray
    loewner_certified: np.ndarray
    loewner_borderline: np.ndarray
    top_affine: np.ndarray
    bottom_affine: np.ndarray
    shared_lattice: np.ndarray
    canonical_keys: np.ndarray
    pareto_optimal: np.ndarray
    rank: np.ndarray
    stats: dict[str, Any]
    raw_twist_radians: np.ndarray
    coincidence_indices: np.ndarray

    @property
    def top_layer_strains(self) -> np.ndarray:
        """Realised principal logarithmic strain applied to the top layer."""

        return self.principal_strains * self.sharing_fraction[:, None]

    @property
    def bottom_layer_strains(self) -> np.ndarray:
        """Realised principal logarithmic strain applied to the bottom layer."""

        return self.principal_strains * (self.sharing_fraction - 1.0)[:, None]

    def __len__(self) -> int:
        return int(self.top_matrices.shape[0])
