"""The defect workflow must be able to reach the interstitial sites that are
saddles of the distance to the nearest atom.

Body-centred cubic iron is the case that matters: its widest hole is the
tetrahedral one, but carbon in ferrite sits at the octahedral site, which is the
midpoint of two second-neighbour atoms and therefore a saddle rather than a
vertex of the Voronoi diagram.  The numbers below -- the two radii, the site
multiplicities 12 and 6, and the position of the inserted atom -- are fixed by
the crystallography of the structure, not by this implementation.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from cellstine.defect.records import DefectAnalysis
from cellstine.defect.workflow import Defect
from cellstine.io import native as io_mod

from conftest import write_poscar

IRON_CONSTANT = 2.87


@pytest.fixture(scope="module")
def iron_bulk_path(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("ferrite") / "fe2.vasp"
    lattice = IRON_CONSTANT * np.eye(3)
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    return str(write_poscar(path, lattice, ["Fe"], [2], positions, comment="bcc iron"))


@pytest.fixture
def workflow(tmp_path) -> Defect:
    return Defect(runs_root=str(tmp_path / "runs"), output_root=str(tmp_path / "output"))


def _interstitials(analysis: DefectAnalysis) -> list:
    return [site for site in analysis.sites if site.site_kind == "interstitial"]


def _nearest_host_distance(lattice: np.ndarray, hosts: np.ndarray, point: np.ndarray) -> float:
    delta = np.asarray(hosts, dtype=float) - np.asarray(point, dtype=float)
    delta -= np.round(delta)
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=float)
    return min(
        float(np.linalg.norm((delta + shift) @ np.asarray(lattice, dtype=float), axis=1).min())
        for shift in shifts
    )


def test_ferrite_reports_only_the_tetrahedral_hole_by_default(workflow, iron_bulk_path):
    analysis = DefectAnalysis.from_dict(workflow.analyse(iron_bulk_path).payload["analysis"])
    sites = _interstitials(analysis)
    assert len(sites) == 1
    assert sites[0].void_radius == pytest.approx(IRON_CONSTANT * math.sqrt(5.0) / 4.0, abs=1e-3)
    assert sites[0].void_kind == "maximum"
    assert sites[0].multiplicity == 12


def test_ferrite_octahedral_site_is_found_with_the_saddles(workflow, iron_bulk_path):
    analysis = DefectAnalysis.from_dict(
        workflow.analyse(iron_bulk_path, interstitial_saddles=True).payload["analysis"]
    )
    octahedral = [
        site
        for site in _interstitials(analysis)
        if abs(float(site.void_radius) - IRON_CONSTANT / 2.0) < 1e-3
    ]
    assert len(octahedral) == 1
    site = octahedral[0]
    assert site.void_kind == "saddle"
    assert site.void_coordination == 2
    assert site.multiplicity == 6
    assert sorted(round(float(value) % 1.0, 6) for value in site.direct) == [0.0, 0.0, 0.5]


def test_carbon_lands_on_the_octahedral_site_of_ferrite(workflow, iron_bulk_path):
    analysis = DefectAnalysis.from_dict(
        workflow.analyse(iron_bulk_path, interstitial_saddles=True).payload["analysis"]
    )
    octahedral = next(
        site
        for site in _interstitials(analysis)
        if abs(float(site.void_radius) - IRON_CONSTANT / 2.0) < 1e-3
    )
    result = workflow.generate(
        iron_bulk_path,
        "interstitial",
        species="C",
        site_ids=[octahedral.site_id],
        interstitial_saddles=True,
    )
    assert result.summary["generated"] == 1

    entry = result.payload["generated"][0]
    defected = io_mod.read_poscar(entry["output_path"])
    assert defected.natoms == 3
    assert dict(zip(defected.species, defected.counts)) == {"Fe": 2, "C": 1}
    positions = np.mod(np.asarray(defected.positions_direct, dtype=float), 1.0)
    inserted = positions[-1]
    assert sorted(round(float(value), 6) for value in inserted) == [0.0, 0.0, 0.5]
    measured = _nearest_host_distance(defected.lattice, positions[:-1], inserted)
    assert measured == pytest.approx(IRON_CONSTANT / 2.0, abs=1e-3)


def test_the_analysis_says_which_kind_of_site_each_interstitial_is(workflow, iron_bulk_path):
    analysis = DefectAnalysis.from_dict(
        workflow.analyse(iron_bulk_path, interstitial_saddles=True).payload["analysis"]
    )
    kinds = {site.void_kind for site in _interstitials(analysis)}
    assert kinds == {"maximum", "saddle"}
    table = workflow.format_analysis(analysis)
    assert "empty sphere" in table
    assert "sad(2)" in table
    assert "max(4)" in table
