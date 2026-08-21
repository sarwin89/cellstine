"""Small text previews for interactive and CLI result summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..moire.search.results import read_results
from .manifests import RunManifest


def _get(item: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(item, dict) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return default


def _matrix(item: Any, prefix: str) -> str:
    if prefix == "i":
        first = _get(item, "layer1_vector1", default=None)
        second = _get(item, "layer1_vector2", default=None)
    else:
        first = _get(item, "layer2_vector1", default=None)
        second = _get(item, "layer2_vector2", default=None)
    if first is None:
        first = [_get(item, f"{prefix}11", default=0), _get(item, f"{prefix}12", default=0)]
    if second is None:
        second = [_get(item, f"{prefix}21", default=0), _get(item, f"{prefix}22", default=0)]
    return f"({int(first[0]):d},{int(first[1]):d}) ({int(second[0]):d},{int(second[1]):d})"


def _percent(value: Any) -> float:
    return 100.0 * float(value or 0.0)


def _format_matrix(value: Any, *, precision: int | None = None) -> str:
    rows = list(value)
    if precision is None:
        return "[" + ", ".join(
            "[" + ", ".join(str(int(item)) for item in row) + "]" for row in rows
        ) + "]"
    return "[" + ", ".join(
        "[" + ", ".join(f"{float(item):.{precision}f}" for item in row) + "]"
        for row in rows
    ) + "]"


def _format_gram_candidates(
    candidates: Sequence[Any], *, limit: int, title: str | None
) -> str:
    rows = sorted(
        list(candidates),
        key=lambda item: (
            float(_get(item, "angle_deg", default=0.0)),
            int(_get(item, "rank", default=0)),
            int(_get(item, "atom_count", default=0)),
        ),
    )
    shown = rows[: max(0, int(limit))]
    if not shown:
        return "No bilayer candidates found."

    lines = []
    if title:
        lines.append(str(title))
    lines.extend(
        [
            " idx  angle (deg)  relative principal strain (%)  top strain (%)  bottom strain (%)  top/bottom/total atoms  rank  Pareto  certification",
            "-" * 142,
        ]
    )
    for candidate in shown:
        strains = list(_get(candidate, "strain", default=(0.0, 0.0)))
        certification = (
            "borderline"
            if bool(_get(candidate, "loewner_borderline", default=False))
            else "certified"
            if bool(_get(candidate, "loewner_certified", default=False))
            else "uncertified"
        )
        lines.append(
            f"{int(_get(candidate, 'index')):4d}  "
            f"{float(_get(candidate, 'angle_deg')):11.4f}  "
            f"({100.0 * float(strains[0]):+9.4f}, {100.0 * float(strains[1]):+9.4f})  "
            f"{_percent(_get(candidate, 'top_strain')):14.4f}  "
            f"{_percent(_get(candidate, 'bottom_strain')):17.4f}  "
            f"{int(_get(candidate, 'top_atom_count')):4d}/"
            f"{int(_get(candidate, 'bottom_atom_count')):6d}/"
            f"{int(_get(candidate, 'atom_count')):5d}  "
            f"{int(_get(candidate, 'rank')):4d}  "
            f"{'yes' if bool(_get(candidate, 'pareto_optimal')) else 'no':>6s}  "
            f"{certification}"
        )
        lines.append(
            "      "
            f"top matrix={_format_matrix(_get(candidate, 'top_matrix'))}; "
            f"bottom matrix={_format_matrix(_get(candidate, 'bottom_matrix'))}; "
            f"shared lattice={_format_matrix(_get(candidate, 'shared_lattice'), precision=6)}"
        )
    if len(rows) > len(shown):
        lines.append(f"... {len(rows) - len(shown)} more candidate(s) not shown.")
    return "\n".join(lines)


def format_bilayer_candidates(candidates: Sequence[Any], *, limit: int = 10, title: str | None = None) -> str:
    """Return a compact table of bilayer candidates in increasing angle order."""

    rows = list(candidates)
    if rows and _get(rows[0], "top_matrix", default=None) is not None:
        return _format_gram_candidates(rows, limit=limit, title=title)
    rows.sort(
        key=lambda item: (
            float(_get(item, "angle_deg", "angle", default=0.0)),
            float(_get(item, "strain_avg", default=0.0)),
            int(_get(item, "total_atoms", "atoms", default=0)),
        )
    )
    shown = rows[: max(0, int(limit))]
    if not shown:
        return "No bilayer candidates found."

    lines = []
    if title:
        lines.append(str(title))
    lines.extend(
        [
            " idx  angle(deg)  strain_avg(%)  strain_1(%)  strain_2(%)   atoms  ratio   aspect  minang  bottom matrix      top matrix",
            "-" * 122,
        ]
    )
    for fallback_index, candidate in enumerate(shown, start=1):
        idx = int(_get(candidate, "index", "idx", default=fallback_index))
        angle = float(_get(candidate, "angle_deg", "angle", default=0.0))
        ratio1 = int(_get(candidate, "ratio1", default=0))
        ratio2 = int(_get(candidate, "ratio2", default=0))
        atoms = int(_get(candidate, "total_atoms", "atoms", default=0))
        aspect = float(_get(candidate, "cell_aspect_ratio", "aspect", default=0.0) or 0.0)
        min_angle = float(_get(candidate, "cell_angle_deg", "min_angle", default=0.0) or 0.0)
        lines.append(
            f"{idx:4d}  {angle:10.4f}  {_percent(_get(candidate, 'strain_avg')):13.4f}  "
            f"{_percent(_get(candidate, 'strain_layer1', 'strain1')):11.4f}  "
            f"{_percent(_get(candidate, 'strain_layer2', 'strain2')):11.4f}  "
            f"{atoms:6d}  {ratio1:3d}/{ratio2:<3d}  {aspect:7.2f}  {min_angle:6.1f}  "
            f"{_matrix(candidate, 'i'):<17s}  {_matrix(candidate, 'j')}"
        )
    if len(rows) > len(shown):
        lines.append(f"... {len(rows) - len(shown)} more candidate(s) not shown.")
    return "\n".join(lines)


def format_nlayer_candidates(candidates: Sequence[Any], *, limit: int = 10, title: str | None = None) -> str:
    """Return a compact table of N-layer candidates in increasing angle order."""

    rows = list(candidates)
    rows.sort(
        key=lambda item: (
            tuple(float(_get(layer, "angle_deg", default=0.0)) for layer in list(_get(item, "upper_layers", default=[]))),
            float(_get(item, "strain_max", default=0.0)),
            float(_get(item, "strain_mean", default=0.0)),
            int(_get(item, "total_atoms", default=0)),
        )
    )
    shown = rows[: max(0, int(limit))]
    if not shown:
        return "No N-layer candidates found."

    lines = []
    if title:
        lines.append(str(title))
    lines.extend(
        [
            " idx  strain_max(%)  strain_mean(%)   atoms  bottom ratio  upper angles (deg)",
            "-" * 80,
        ]
    )
    for fallback_index, candidate in enumerate(shown, start=1):
        idx = int(_get(candidate, "index", default=fallback_index))
        layers = list(_get(candidate, "upper_layers", default=[]))
        angle_summary = ", ".join(
            f"L{int(_get(layer, 'layer_index', default=0)) + 1}={float(_get(layer, 'angle_deg', default=0.0)):.3f}"
            for layer in layers
        )
        lines.append(
            f"{idx:4d}  {_percent(_get(candidate, 'strain_max')):13.4f}  "
            f"{_percent(_get(candidate, 'strain_mean')):14.4f}  "
            f"{int(_get(candidate, 'total_atoms', default=0)):6d}  "
            f"{int(_get(candidate, 'ratio_bottom', default=0)):12d}  {angle_summary}"
        )
    if len(rows) > len(shown):
        lines.append(f"... {len(rows) - len(shown)} more candidate(s) not shown.")
    return "\n".join(lines)


def _iter_manifest_result_paths(path: Path) -> Iterable[Path]:
    manifest = RunManifest.load(path)
    priority = ["results_json", "results_dat"]
    for key in priority:
        value = manifest.artifacts.get(key)
        if value is not None:
            yield Path(str(value)).resolve()
    for key, value in sorted(manifest.artifacts.items()):
        if key in priority:
            continue
        if key.startswith("results_dat") or key.startswith("results_json"):
            yield Path(str(value)).resolve()


def preview_moire_results_file(path: str | Path, *, limit: int = 15) -> str:
    """Preview validated native Gram JSON, rejecting positional DAT results."""

    source = Path(path).resolve()
    result_paths = list(_iter_manifest_result_paths(source)) if source.name == "manifest.json" else [source]
    sections = []
    for result_path in result_paths:
        payload = read_results(result_path)
        search = payload["search"]
        metadata = payload["metadata"]
        fallback = metadata["symmetric_fallback"] or "none"
        title = (
            f"Saved Gram candidates: {result_path.name}\n"
            f"schema={payload['schema']} v{payload['version']}; "
            f"engine={metadata['engine']}; max length={float(search['max_length']):g} Angstrom; "
            f"top strain={float(search['top_strain']):g}; "
            f"bottom strain={float(search['bottom_strain']):g}; "
            f"symmetric requested={metadata['symmetric_requested']}; "
            f"symmetric used={metadata['symmetric_used']}; "
            f"symmetric fallback={fallback}; "
            f"stage stats={json.dumps(metadata['stage_stats'], sort_keys=True)}"
        )
        sections.append(
            format_bilayer_candidates(payload["candidates"], limit=limit, title=title)
        )
    return "\n\n".join(section for section in sections if section) or "No saved candidates could be previewed."


def format_adsorption_sites(sites: Sequence[Any], *, limit: int = 30, title: str | None = None) -> str:
    """Return a compact table of adsorption site coordinates."""

    rows = list(sites)
    rows.sort(key=lambda item: tuple(float(value) for value in _get(item, "direct", default=(0.0, 0.0, 0.0))))
    shown = rows[: max(0, int(limit))]
    if not shown:
        return "No adsorption sites found."
    lines = []
    if title:
        lines.append(str(title))
    lines.extend(
        [
            " idx  type              direct (u, v, w)              cartesian (x, y, z) Ang",
            "-" * 82,
        ]
    )
    for index, site in enumerate(shown, start=1):
        direct = tuple(float(value) for value in _get(site, "direct", default=(0.0, 0.0, 0.0)))
        cartesian = tuple(float(value) for value in _get(site, "cartesian", default=(0.0, 0.0, 0.0)))
        site_type = str(_get(site, "site_type", default="site"))
        lines.append(
            f"{index:4d}  {site_type:<16s}  "
            f"({direct[0]:7.4f}, {direct[1]:7.4f}, {direct[2]:7.4f})    "
            f"({cartesian[0]:9.4f}, {cartesian[1]:9.4f}, {cartesian[2]:9.4f})"
        )
    if len(rows) > len(shown):
        lines.append(f"... {len(rows) - len(shown)} more site(s) not shown.")
    return "\n".join(lines)


def format_site_report(site_report: Any, *, limit_per_type: int = 12) -> str:
    """Return grouped adsorption-site previews for a site-analysis result."""

    grouped: dict[str, list[Any]] = {}
    for site in list(_get(site_report, "sites", default=[])):
        grouped.setdefault(str(_get(site, "site_type", default="site")), []).append(site)
    if not grouped:
        return "No adsorption sites found."
    sections = []
    for site_type in sorted(grouped):
        sections.append(
            format_adsorption_sites(
                grouped[site_type],
                limit=limit_per_type,
                title=f"{site_type} sites ({len(grouped[site_type])} found)",
            )
        )
    return "\n\n".join(sections)
