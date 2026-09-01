"""Checks for serializable symmetry workflow models."""

from __future__ import annotations

from cellstine.symmetry.models import EquivalentAtomGroup, SymmetryAnalysis, SymmetryOperation


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
