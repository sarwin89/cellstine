"""The moire workflow must stay symmetry-aware for realistically printed cells.

A POSCAR written with the customary six decimal places is not exactly hexagonal.
Detecting layer symmetry at machine precision therefore throws it away, and the
search then reports every symmetry image of the same bilayer as a separate
candidate and attaches a spurious anisotropic strain to it.  These tests pin the
tolerant detection plus exact idealisation that keeps such runs clean.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from cellstine.core.symmetry2d import DEFAULT_SYMMETRY_TOLERANCE
from cellstine.moire.search.find import run_find

from conftest import write_poscar


def _rounded_honeycomb(
    path: Path, constant: float, species: list[str], decimals: int = 6
) -> Path:
    """Write a honeycomb POSCAR whose lattice is rounded like a printed file."""

    height = round(constant * math.sqrt(3.0) / 2.0, decimals)
    lattice = np.array(
        [
            [round(constant, decimals), 0.0, 0.0],
            [-round(0.5 * constant, decimals), height, 0.0],
            [0.0, 0.0, 20.0],
        ]
    )
    positions = np.array([[0.0, 0.0, 0.5], [1.0 / 3.0, 2.0 / 3.0, 0.5]])
    if species[0] == species[1]:
        return write_poscar(path, lattice, [species[0]], [2], positions)
    return write_poscar(path, lattice, species, [1, 1], positions)


@pytest.fixture(scope="module")
def rounded_pair(tmp_path_factory) -> tuple[Path, Path]:
    directory = tmp_path_factory.mktemp("rounded")
    top = _rounded_honeycomb(directory / "graphene.vasp", 2.468, ["C", "C"])
    bottom = _rounded_honeycomb(directory / "hbn.vasp", 2.504, ["B", "N"])
    return top, bottom


def _run(top: Path, bottom: Path, output_root: Path, **kwargs):
    return run_find(
        top_poscar=str(top),
        bottom_poscar=str(bottom),
        max_length=16.0,
        top_strain=0.01,
        bottom_strain=0.01,
        output_root=str(output_root),
        **kwargs,
    )


def test_rounded_layers_keep_their_rotation_orders(rounded_pair, tmp_path):
    top, bottom = rounded_pair
    run = _run(top, bottom, tmp_path / "default")
    assert run.parameters["top_rotation_order"] == 6
    assert run.parameters["bottom_rotation_order"] == 3
    assert run.parameters["angle_period_deg"] == pytest.approx(60.0)


def test_tolerant_detection_removes_symmetry_duplicates(rounded_pair, tmp_path):
    """Machine-precision detection reports the same bilayer many times over."""

    top, bottom = rounded_pair
    tolerant = _run(top, bottom, tmp_path / "tolerant")
    strict = _run(top, bottom, tmp_path / "strict", symmetry_tolerance=1e-9)
    assert strict.parameters["top_rotation_order"] == 2
    assert len(tolerant.candidates) < len(strict.candidates)
    tolerant_angles = sorted(
        round(candidate["angle_deg"], 4) for candidate in tolerant.candidates
    )
    assert len(set(tolerant_angles)) == len(tolerant_angles)
    for candidate in tolerant.candidates:
        assert abs(candidate["angle_deg"]) <= 30.0 + 1e-6


def test_idealisation_removes_the_spurious_anisotropic_strain(rounded_pair, tmp_path):
    """Two isotropic layers must give two equal principal strains."""

    top, bottom = rounded_pair
    run = _run(top, bottom, tmp_path / "isotropic")
    aligned = min(run.candidates, key=lambda item: abs(item["angle_deg"]))
    assert aligned["strain"][0] == pytest.approx(aligned["strain"][1], abs=1e-12)
    expected = math.log(2.504 / 2.468)
    assert aligned["strain"][0] == pytest.approx(expected, rel=1e-5)


def test_idealisation_is_recorded_and_small(rounded_pair, tmp_path):
    import json

    top, bottom = rounded_pair
    run = _run(top, bottom, tmp_path / "recorded")
    document = json.loads(Path(run.result_path).read_text())
    metadata = document["metadata"]
    assert metadata["symmetry_tolerance"] == pytest.approx(DEFAULT_SYMMETRY_TOLERANCE)
    for name in ("top_idealisation", "bottom_idealisation"):
        assert 0.0 <= metadata[name] < DEFAULT_SYMMETRY_TOLERANCE


def test_shared_moire_cell_is_exactly_equilateral(rounded_pair, tmp_path):
    top, bottom = rounded_pair
    run = _run(top, bottom, tmp_path / "cells")
    for candidate in run.candidates:
        lattice = np.asarray(candidate["shared_lattice"], dtype=float)
        lengths = np.linalg.norm(lattice, axis=0)
        if abs(candidate["moire_gamma_deg"] - 120.0) > 1e-6:
            continue
        assert lengths[0] == pytest.approx(lengths[1], rel=1e-12)
