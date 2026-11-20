"""Interface contact metadata follows physical boundary positions."""

import json
import numpy as np
import pytest

from cellstine.interface.surface import stacking
from cellstine.interface.workflow import assembly
from cellstine.interface.workflow.interface import Interface
from cellstine.io.converters import StructureConverter
from cellstine.io.models import StructureRecord
from cellstine.io.vasp import VaspIO


def _slab(angle, offset=0, origin=(0., 0.)):
    cell = np.array([[1., 0., 0.],
        [.5 if angle == 60 else -.5, np.sqrt(3.) / 2., 0.], [0., 0., 12.]])
    hollow = np.array([1. / 3., 1. / 3. if angle == 60 else 2. / 3.])
    points = np.array([[*((layer + offset) * hollow + origin), .2 + .2 * layer]
                       for layer in range(3)])
    return StructureRecord(comment="registry", lattice=cell, species=["Al"], counts=[3],
        positions_direct=points, positions_cartesian=points @ cell)


@pytest.mark.parametrize("angle", [60, 120])
@pytest.mark.parametrize("offset,delta", [(0, 1), (1, 2), (2, 0)])
@pytest.mark.parametrize("origin", [(0., 0.), (.17, -.23), (2., -3.)])
def test_passive_contact_uses_relative_boundary_positions(angle, offset, delta, origin):
    bottom, top = _slab(angle, origin=origin), _slab(angle, offset, origin)
    summary = assembly.realised_stacking_summary(bottom, top)
    assert summary["delta"] == delta
    assert summary["registry_requested"] is False
    _, unchanged, passive = assembly.prepare_stacking(bottom, top,
        bottom_stacking="keep", top_stacking="keep", registry=None)
    assert unchanged is top and passive == summary


def test_nonideal_contact_is_unclassified_without_moving_the_slabs():
    bottom, top = _slab(60), _slab(60, origin=(.13, .07))
    saved = top.positions_direct.copy()
    _, output, summary = assembly.prepare_stacking(bottom, top,
        bottom_stacking="keep", top_stacking="abc", registry=None)
    assert output is top and np.array_equal(top.positions_direct, saved)
    assert summary["delta"] is None
    assert summary["contact"] == "unclassified"
    assert summary["kind"] == "general_offset"
    assert summary["registry_requested"] is False


@pytest.mark.parametrize("registry", ["fcc", "hcp", "eclipsed"])
def test_requested_registry_is_realized_after_translated_input(registry):
    bottom, top, summary = assembly.prepare_stacking(_slab(60), _slab(60, 2),
        bottom_stacking="keep", top_stacking="abc", registry=registry)
    # Actual reporting must agree after the requested physical translation.
    assert assembly.realised_stacking_summary(bottom, top)["delta"] == summary["delta"]
    assert summary["registry_requested"] is True


@pytest.mark.parametrize("displacement,expected", [(.049, 0), (.051, None)])
def test_registry_uses_existing_componentwise_physical_tolerance(displacement, expected):
    bottom, top = _slab(60), _slab(60, 2)
    top.positions_cartesian[:, 0] += displacement
    top.positions_direct = top.positions_cartesian @ np.linalg.inv(top.lattice)
    assert assembly.realised_stacking_summary(bottom, top)["delta"] == expected


def test_public_label_only_summary_preserves_its_legacy_inference():
    bottom = stacking.analyse_stacking(_slab(60))
    top = stacking.analyse_stacking(_slab(60, 2), hollow_cartesian=bottom.hollow_cartesian)
    summary = assembly.contact_summary(bottom, top, delta=None,
        bottom_mirrored=False, top_mirrored=False)
    assert summary["delta"] == 1 and summary["contact"] == "C-A"
    assert list(summary) == ["bottom_sequence", "top_sequence", "bottom_sense",
        "top_sense", "bottom_mirrored", "top_mirrored", "delta", "contact", "kind",
        "registry_requested"]


@pytest.mark.parametrize("origin,expected", [((1. / 3., 1. / 3.), 2), ((.13, .07), None)])
def test_written_interface_and_manifest_preserve_measured_contact(tmp_path, origin, expected):
    bottom, top = _slab(60), _slab(60, origin=origin)
    paths = [tmp_path / "bottom.vasp", tmp_path / "top.vasp"]
    for record, path in zip((bottom, top), paths):
        VaspIO().write(record, str(path), positions_are_cartesian=False, wrap_positions=True)
    destination = tmp_path / "joined.vasp"
    workflow = Interface(backend="native", runs_root=tmp_path / "runs", output_root=tmp_path / "output")
    result = workflow.build(bottom_input=str(paths[0]), top_input=str(paths[1]),
        bottom_kind="slab", top_kind="slab", gap=2.4, vacuum=12., output_path=str(destination))
    assert result.summary["stacking"]["delta"] == expected
    manifests = list((tmp_path / "runs").rglob("manifest.json"))
    assert manifests
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["summary"]["stacking"] == result.summary["stacking"]
    output = StructureConverter().read(str(destination))
    cartesian = output.positions_cartesian
    # The contacting atoms are bottom index2 and top index3. Their planar
    # displacement is unchanged by vertical placement and cell wrapping.
    actual = (cartesian[3] - cartesian[2])[:2]
    incoming = (top.positions_cartesian[0] - bottom.positions_cartesian[2])[:2]
    coefficients = (actual - incoming) @ np.linalg.inv(bottom.lattice[:2, :2])
    assert np.allclose(coefficients, np.rint(coefficients), atol=1e-10)
