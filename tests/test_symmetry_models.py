"""Checks for serializable symmetry workflow models."""

from __future__ import annotations

from cellstine.symmetry.models import EquivalentAtomGroup, SymmetryAnalysis, SymmetryOperation
from cellstine.symmetry.reporting import format_symmetry_analysis


def test_symmetry_analysis_model_serializes_the_public_schema():
    analysis = SymmetryAnalysis(
        structure_path="cell.vasp",
        backend="native",
        atom_count=2,
        species=["Si"],
        counts=[2],
        lattice_parameters={"a": 1.0, "b": 1.0, "c": 1.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0, "volume": 1.0},
        operations=[SymmetryOperation(rotation=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], translation=(0.0, 0.0, 0.0))],
        equivalent_groups=[
            EquivalentAtomGroup(
                group_id="Si1",
                species="Si",
                representative_index=1,
                equivalent_indices=[1, 2],
                multiplicity=2,
                wyckoff="a",
            )
        ],
        origin_shift=(0.0, 0.5, 0.0),
    )

    payload = analysis.to_dict()

    assert payload["schema"] == "cellstine.symmetry_analysis.v1"
    assert payload["operations"] == [{"rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "translation": (0.0, 0.0, 0.0)}]
    assert payload["equivalent_groups"][0]["equivalent_indices"] == [1, 2]
    assert payload["origin_shift"] == [0.0, 0.5, 0.0]


def test_symmetry_analysis_formatter_keeps_the_cli_preview_contract():
    analysis = SymmetryAnalysis(
        structure_path=None,
        backend="native",
        atom_count=3,
        species=["Mo", "S"],
        counts=[1, 2],
        lattice_parameters={"a": 3.16, "b": 3.16, "c": 20.0, "alpha": 90.0, "beta": 90.0, "gamma": 120.0, "volume": 172.8},
        point_group="6mm",
        crystal_system="hexagonal",
        lattice_point_group="6/mmm",
        operation_count=12,
        centering_translation_count=1,
        equivalent_groups=[
            EquivalentAtomGroup("Mo1", "Mo", 1, [1], 1),
            EquivalentAtomGroup("S1", "S", 2, [2, 3], 2, wyckoff="e"),
        ],
        notes=["native symmetry engine"],
    )

    preview = format_symmetry_analysis(analysis)

    assert preview.startswith("Symmetry analysis (native)")
    assert "Point group: 6mm" in preview
    assert "Lattice point group: 6/mmm" in preview
    assert "S1 S mult=2 wyckoff=e atoms=2,3" in preview
    assert "- native symmetry engine" in preview
