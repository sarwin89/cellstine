"""Workflow metadata describes the requested enumeration and displayed rows."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cellstine.defect.records import DefectAnalysis, DefectSite
from cellstine.defect.workflow import Defect
from cellstine.io import native as io_mod

from conftest import write_poscar


@pytest.fixture
def saved_analysis(tmp_path):
    source = write_poscar(tmp_path / "host.vasp", 3 * np.eye(3), ["C"], [1], np.zeros((1, 3)))
    site = DefectSite("atom_001", "C", None, (0., 0., 0.), (0., 0., 0.),
                      equivalent_indices=[1], representative_index=1)
    analysis = DefectAnalysis(str(source), "bulk", "native", 1, ["C"], [1], [], [site])
    saved = tmp_path / "analysis.json"
    saved.write_text(json.dumps(analysis.to_dict()), encoding="utf-8")
    workflow = Defect(runs_root=str(tmp_path / "runs"), output_root=str(tmp_path / "output"))
    return workflow, source, saved, analysis


@pytest.mark.parametrize("limit,shown", [(-5, 0), (0, 0), (1, 1), (30, 1)])
def test_preview_summary_agrees_with_clamped_display_limit(saved_analysis, limit, shown):
    workflow, _, saved, _ = saved_analysis
    result = workflow.preview(str(saved), limit=limit)
    assert result.summary["shown_sites"] == shown
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["summary"]["shown_sites"] == shown
    assert manifest["parameters"]["limit"] == limit
    assert result.summary["total_sites"] == 1


@pytest.mark.parametrize("stage", ["analyse", "preview", "generate"])
@pytest.mark.parametrize("saddles", [False, True, np.bool_(True)])
def test_manifest_records_requested_saddles_even_when_reusing_saved_analysis(
    saved_analysis, stage, saddles, monkeypatch
):
    workflow, source, saved, analysis = saved_analysis
    if stage == "analyse":
        # The physical saddle enumeration is covered by ferrite tests; isolate
        # requested-parameter serialization from that expensive enumeration.
        monkeypatch.setattr(workflow, "_analyse_record", lambda *args, **kwargs: analysis)
        result = workflow.analyse(str(source), interstitial_saddles=saddles)
    elif stage == "preview":
        result = workflow.preview(str(saved), interstitial_saddles=saddles)
    else:
        result = workflow.generate(str(saved), "vacancy", site_ids=["atom_001"], interstitial_saddles=saddles)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["parameters"]["interstitial_saddles"] is bool(saddles)


@pytest.mark.parametrize("preserve", [False, True, np.bool_(True)])
def test_generate_manifest_records_requested_vacuum_policy(saved_analysis, preserve):
    workflow, _, saved, _ = saved_analysis
    result = workflow.generate(str(saved), "vacancy", site_ids=["atom_001"], preserve_vacuum=preserve)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["parameters"]["preserve_vacuum"] is bool(preserve)


@pytest.mark.parametrize("tokens,removed", [(["atom_001", "atom_002"], [1, 3]),
                                           (["1", "2"], [1, 2]),
                                           (["atom_1", "atom_2"], [1, 2])])
def test_manual_divacancy_resolves_displayed_site_ids_before_numeric_aliases(tmp_path, tokens, removed):
    lattice = 8 * np.eye(3)
    direct = np.array([[0., 0., 0.], [.2, 0., 0.], [.4, 0., 0.], [.6, 0., 0.]])
    source = write_poscar(tmp_path / "host.vasp", lattice, ["C"], [4], direct)
    sites = [DefectSite(f"atom_{number:03d}", "C", None, tuple(direct[index - 1]),
                       tuple(direct[index - 1] @ lattice), equivalent_indices=members,
                       representative_index=index, multiplicity=2)
             for number, index, members in [(1, 1, [1, 2]), (2, 3, [3, 4])]]
    analysis = DefectAnalysis(str(source), "bulk", "native", 4, ["C"], [4], [], sites)
    saved = tmp_path / "analysis.json"
    saved.write_text(json.dumps(analysis.to_dict()), encoding="utf-8")
    workflow = Defect(runs_root=str(tmp_path / "runs"), output_root=str(tmp_path / "output"))
    result = workflow.generate(str(saved), "divacancy", site_ids=tokens)
    written = io_mod.read_poscar(result.payload["generated"][0]["output_path"])
    np.testing.assert_allclose(written.positions_direct, direct[[i for i in range(4) if i + 1 not in removed]], atol=1e-10)
