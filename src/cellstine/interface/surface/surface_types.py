"""Shared dataclasses and constants for the surface package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from ...io import native as io_mod


DEFAULT_OUTPUT_DIR = Path("output")


@dataclass
class AdsorptionSite:
    site_type: str
    direct: tuple[float, float, float]
    cartesian: tuple[float, float, float]
    coordination: int | None = None
    void_radius: float | None = None
    subsurface_depth: int | None = None


@dataclass
class SiteAnalysisRun:
    output_path: Path | None
    source_poscar: str | None
    surface_side: str
    top_layer_atom_count: int
    detected_layer_count: int
    nearest_neighbor_distance: float
    neighbor_cutoff: float
    average_top_layer_coordination: float
    site_counts: Dict[str, int]
    sites: List[AdsorptionSite]


@dataclass
class SurfaceRun:
    output_path: Path
    miller: tuple[int, int, int]
    layers: int
    vacuum: float
    total_atoms: int
    repeat_a: int
    repeat_b: int
    supercell_matrix: tuple[int, int, int, int] | None
    site_output_path: Path | None
    site_counts: Dict[str, int] | None


@dataclass(frozen=True)
class PrimitiveSurfaceAnalysis:
    miller: tuple[int, int, int]
    centering: str
    probe_layers: int
    atoms_per_layer: tuple[int, ...]
    stacking_sequence: str
    stacking_period: str
    inplane_angle_deg: float
    lattice: np.ndarray


@dataclass(frozen=True)
class SurfaceStructureBuild:
    structure: io_mod.PoscarData
    repeat_a: int
    repeat_b: int
    supercell_matrix: tuple[int, int, int, int] | None


SITE_TYPE_ALIASES = {
    "top": "top",
    "bridge": "bridge",
    "hcp": "hcp_hollow",
    "hcp_hollow": "hcp_hollow",
    "fcc": "fcc_hollow",
    "fcc_hollow": "fcc_hollow",
    "hollow": "hollow",
    "fourfold": "fourfold_hollow",
    "fourfold_hollow": "fourfold_hollow",
}
