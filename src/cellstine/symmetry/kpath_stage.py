"""The ``kpath`` stage: a band-structure path through the Brillouin zone.

``cellstine symmetry kpath STRUCTURE`` writes the line-mode ``KPOINTS`` file a
band structure is run from, together with a ``kpath.json`` that records where
every high-symmetry point is, how long every segment is, and where the tick
marks of the band plot fall.

The points themselves are not looked up in a table: they are the
zero-dimensional strata of the symmetry of the crystal, and the ends of its
symmetry lines, derived in :mod:`cellstine.core.kpath` from the Brillouin zone
of :mod:`cellstine.core.brillouin` and the Bravais classification of
:mod:`cellstine.core.bravais`.  Only the *names*, and the order of the visits
for the Bravais types that have a conventional one, come from convention.

The stage is a mixin so that any workflow class carrying a structure converter
can offer it; it is mixed into the symmetry workflow, which is where the
reciprocal-space commands live.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..core import symmetry3d
from ..core.base import run_output_suffix
from ..core.bravais import conventional_cell
from ..core.kpath import BandPath, band_path
from ..core.models import CommandResult
from ..core.species import expand_species
from ..io import kpoints as kpoints_io

#: A segment shorter than this fraction of the longest one is sampled far more
#: finely than asked for, because a line-mode file carries one division count.
SHORT_SEGMENT_FRACTION = 0.05


def format_kpath_table(path: BandPath) -> str:
    """Return the table of high-symmetry points of a band path."""

    header = (
        f"{'point':>8}  {'fractional coordinates':>34}  {'|k| (1/A)':>10}  "
        f"{'little group':>12}  {'kind':>7}  name"
    )
    lines = [header, "-" * len(header)]
    kinds = {0: "point", 1: "line", 2: "plane"}
    for item in path.points:
        coordinates = "  ".join(f"{value:9.5f}" for value in item.fractional)
        name = "derived" if item.derived_label else "standard"
        if item.aliases:
            name = f"{name} (also {', '.join(item.aliases)})"
        lines.append(
            f"{item.label:>8}  {coordinates:>34}  {item.length:10.5f}  "
            f"{item.little_group_order:12d}  {kinds.get(item.stratum_dimension, '?'):>7}  {name}"
        )
    return "\n".join(lines)


def format_segment_table(path: BandPath) -> str:
    """Return the table of segments of a band path."""

    header = f"{'segment':>18}  {'length (1/A)':>12}  {'from (1/A)':>10}  along"
    lines = [header, "-" * len(header)]
    travelled = 0.0
    lengths = path.segment_lengths()
    kinds = {1: "symmetry line", 2: "mirror plane"}
    for (start, end), length, dimension in zip(path.segments, lengths, path.segment_strata):
        along = kinds.get(dimension, "straight line")
        lines.append(f"{start + ' -> ' + end:>18}  {length:12.5f}  {travelled:10.5f}  {along}")
        travelled += length
    return "\n".join(lines)


class BandPathMixin:
    """The ``kpath`` stage: high-symmetry points and a line-mode k-point file."""

    def kpath(
        self,
        structure_path: str,
        *,
        spacing: float | None = 0.03,
        divisions: int | None = None,
        path: str | None = None,
        use_standard: bool = True,
        use_symmetry: bool = True,
        time_reversal: bool = True,
        symprec: float = 0.01,
        output_path: str | Path | None = None,
    ) -> CommandResult:
        """Write a band-structure path and its line-mode k-point file.

        Either ``spacing`` -- a largest step along the path, in inverse angstrom
        and in the ``2 pi`` convention -- or an explicit ``divisions`` count per
        segment fixes how finely the path is sampled.  ``path`` names the walk
        explicitly, as ``"GAMMA-X-W|K-L"``; otherwise the conventional walk of
        the Bravais type is used when there is one, and one derived from the
        symmetry lines when there is not.

        The symmetry used is that of the decorated cell, so a crystal whose
        atoms break some of the symmetry of its lattice gets the points its own
        group singles out; ``use_symmetry=False`` uses the point group of the
        lattice alone.
        """

        source = str(Path(structure_path).resolve())
        record = self.converter.read(source, canonicalize=False)
        lattice = np.asarray(record.lattice, dtype=float)

        rotations = None
        crystal_operations = None
        centering_count = None
        if use_symmetry:
            rotations, operation_translations = symmetry3d.symmetry_operations(
                lattice,
                np.asarray(record.positions_direct, dtype=float),
                expand_species(record.species, record.counts),
                symprec=float(symprec),
            )
            crystal_operations = int(len(rotations))
            centering_count = int(
                len(symmetry3d.pure_translations(rotations, operation_translations))
            )
        lattice_operations = int(len(symmetry3d.lattice_point_group(lattice)))

        result = band_path(
            lattice,
            rotations,
            time_reversal=bool(time_reversal),
            path=path,
            use_standard=bool(use_standard),
        )
        cell = conventional_cell(lattice)

        if divisions is None and spacing is None:
            raise ValueError("give either a k-point spacing along the path or explicit divisions")
        if divisions is not None and spacing is not None:
            raise ValueError("give a k-point spacing or explicit divisions, not both")
        count = int(divisions) if divisions is not None else result.divisions_for_spacing(float(spacing))
        if count < 2:
            raise ValueError("a segment of the path needs at least two points")

        run_id, run_dir = self.create_run_dir("kpath", label=Path(source).stem)
        destination = (
            Path(output_path).resolve()
            if output_path is not None
            else self.output_root
            / f"KPOINTS_band_{Path(source).stem}_{run_output_suffix(run_id).replace('_', '-')}"
        )
        comment = f"{Path(source).stem} {cell.symbol} band path {result.path_string()}"
        written = kpoints_io.write_band_path(destination, result, divisions=count, comment=comment)

        points, distances, labels = result.sample(count)
        ticks = [
            {"label": label, "distance": float(distance)}
            for label, distance in zip(labels, distances)
            if label
        ]
        payload_json: dict[str, Any] = {
            "schema": "cellstine.symmetry.kpath",
            "version": 1,
            "structure": source,
            "divisions": count,
            "spacing": None if spacing is None else float(spacing),
            "sampled_point_count": int(len(points)),
            "conventional_cell": cell.summary(),
            "ticks": ticks,
            **result.summary(),
        }
        kpath_json = run_dir / "kpath.json"
        with kpath_json.open("w", encoding="utf-8") as handle:
            json.dump(payload_json, handle, indent=2)

        lengths = result.segment_lengths()
        warnings: list[str] = []
        if result.path_source == "derived":
            warnings.append(
                f"the {cell.symbol} zone has no conventional band path in CELLSTINE, so the walk was "
                "derived from the symmetry lines of the zone; pass --path to choose your own"
            )
        derived_names = [item.label for item in result.points if item.derived_label]
        if derived_names:
            warnings.append(
                f"{len(derived_names)} point(s) of this zone carry derived names "
                f"({', '.join(derived_names)}): their coordinates are exact, only the letters are "
                "CELLSTINE's own"
            )
        if lengths and min(lengths) < SHORT_SEGMENT_FRACTION * max(lengths):
            warnings.append(
                f"the shortest segment is {min(lengths):.4f} 1/A against {max(lengths):.4f} 1/A for the "
                f"longest; a line-mode file samples every segment with the same {count} points, so the "
                "short one is sampled much more finely than asked for"
            )
        if centering_count is not None and centering_count > 1:
            warnings.append(
                f"the cell is {centering_count}-fold non-primitive, so its zone is "
                f"{centering_count} times smaller than the zone of the primitive cell and every band "
                f"is folded {centering_count} times onto this path; "
                "`cellstine symmetry reduce --cell primitive` writes the primitive cell"
            )
        if crystal_operations is not None and crystal_operations < lattice_operations:
            warnings.append(
                f"the atoms keep {crystal_operations} of the {lattice_operations} operations of the "
                "lattice, so the path follows the symmetry of the crystal rather than the shape of the "
                "zone alone"
            )

        summary: dict[str, Any] = {
            "structure": source,
            "bravais_symbol": cell.symbol,
            "crystal_system": cell.system,
            "path": result.path_string(),
            "path_source": result.path_source,
            "high_symmetry_points": len(result.points),
            "segments": len(result.segments),
            "divisions_per_segment": count,
            "sampled_points": int(len(points)),
            "path_length_inv_ang": round(float(result.length), 6),
            "longest_step_inv_ang": round(float(max(lengths) / (count - 1)) if lengths else 0.0, 6),
            "zone_volume_inv_ang3": round(float(result.zone.volume), 6),
            "time_reversal": bool(time_reversal),
        }
        if warnings:
            summary["warnings"] = warnings

        artifacts = {"kpoints": str(written), "kpath_json": str(kpath_json.resolve())}
        manifest_path = self.write_manifest(
            stage="kpath",
            run_id=run_id,
            run_dir=run_dir,
            backend="native",
            inputs={"structure_path": source},
            parameters={
                "spacing": None if spacing is None else float(spacing),
                "divisions": None if divisions is None else int(divisions),
                "path": None if path is None else str(path),
                "use_standard": bool(use_standard),
                "use_symmetry": bool(use_symmetry),
                "time_reversal": bool(time_reversal),
                "symprec": float(symprec),
            },
            artifacts=artifacts,
            summary=summary,
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts=artifacts,
            summary=summary,
            payload={
                "kpath_preview": format_kpath_table(result),
                "segment_preview": format_segment_table(result),
                "points": [item.summary() for item in result.points],
            },
        )
