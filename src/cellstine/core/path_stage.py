"""The ``path`` stage: migration-path images between two structures of one cell.

``cellstine defect path START END --images N`` -- and the same stage under
``adsorbate`` -- writes the chain of structures a nudged-elastic-band run starts
from: ``00/POSCAR`` is the initial state, the last folder the final one, and the
images in between are evenly spaced along the straight line that joins them in
configuration space.

The two decisions that make the chain meaningful -- which atom of the final
structure each atom of the initial one becomes, and which periodic image it
travels to -- are taken in :mod:`cellstine.core.pathway`, exactly and with a
certificate, rather than by trusting the order the atoms happen to be listed in.

The stage is mixed into any workflow class that carries a structure converter
and a VASP writer, so the same code serves a defect hop and an adsorbate
diffusing between two sites.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .geometry import shortest_lattice_vector_length
from .models import CommandResult
from .pathway import MigrationPath, build_migration_path
from .provenance import restage_comment
from .species import expand_species
from .validation import structure_errors
from ..io.models import StructureRecord

#: An intermediate image whose closest contact falls below this fraction of the
#: closest contact of the endpoints is reported: a straight line between two
#: relaxed structures can push two atoms through each other, and that is the
#: one failure of a linear chain a user has to be told about.
CONTACT_WARNING_FRACTION = 0.75

#: A contact shorter than this is too short whatever the endpoints look like.
CONTACT_FLOOR = 1.0


def format_path_table(path: MigrationPath, image_paths: Sequence[Path]) -> str:
    """Return the per-image table of the chain: how far along, and how close."""

    spacings = path.spacings()
    contacts = path.shortest_contacts()
    travelled = np.concatenate(([0.0], np.cumsum(spacings)))
    header = (
        f"{'image':>6}  {'reaction coordinate':>19}  {'travelled (A)':>13}  "
        f"{'step (A)':>9}  {'closest contact (A)':>19}  file"
    )
    lines = [header, "-" * len(header)]
    total = float(travelled[-1])
    for index, contact in enumerate(contacts):
        fraction = float(index) / float(path.intermediate_count + 1)
        step = f"{spacings[index - 1]:9.4f}" if index else f"{'-':>9}"
        name = str(image_paths[index]) if index < len(image_paths) else "-"
        lines.append(
            f"{index:>6}  {fraction:>19.4f}  {travelled[index]:>13.4f}  {step}  "
            f"{contact:>19.4f}  {name}"
        )
    lines.append("")
    lines.append(f"path length {total:.4f} A over {path.intermediate_count} intermediate image(s)")
    return "\n".join(lines)


def format_moved_atoms(path: MigrationPath, species_by_atom: Sequence[str], limit: int = 12) -> str:
    """Return the atoms that actually move, longest displacement first."""

    distances = path.matching.distances
    order = np.argsort(-distances)
    moving = [index for index in order.tolist() if distances[index] > 1e-8]
    if not moving:
        return ""
    header = f"{'atom':>6}  {'species':>7}  {'displacement (A)':>16}  {'paired with':>11}"
    lines = [header, "-" * len(header)]
    for index in moving[: int(limit)]:
        partner = int(path.matching.partners[index]) + 1
        lines.append(
            f"{index + 1:>6}  {str(species_by_atom[index]):>7}  {distances[index]:>16.4f}  {partner:>11}"
        )
    if len(moving) > int(limit):
        lines.append(f"... and {len(moving) - int(limit)} more moving atom(s)")
    return "\n".join(lines)


def _path_payload(
    path: MigrationPath,
    species_by_atom: Sequence[str],
    image_paths: Sequence[Path],
) -> dict[str, Any]:
    distances = path.matching.distances
    return {
        "schema": "cellstine.defect.path",
        "version": 1,
        "lattice": np.asarray(path.lattice, dtype=float).tolist(),
        "species": list(path.species),
        "counts": [int(value) for value in path.counts],
        "intermediate_images": int(path.intermediate_count),
        "image_count": int(path.image_count),
        "path_length_ang": float(path.path_length),
        "image_spacing_ang": float(path.image_spacing),
        "measured_spacings_ang": [float(value) for value in path.spacings()],
        "shortest_contacts_ang": [float(value) for value in path.shortest_contacts()],
        "maximum_atom_displacement_ang": float(path.maximum_atom_displacement),
        "moved_atom_count": int(path.moved_atom_count),
        "matching": {
            "reordered": not bool(path.matching.identity),
            "partners": [int(value) + 1 for value in path.matching.partners],
            "displacements_ang": [float(value) for value in distances],
            "total_squared_cost_ang2": float(path.matching.cost),
            "certificate_error": float(path.matching.certificate_error),
        },
        "images": [
            {
                "index": index,
                "reaction_coordinate": float(index) / float(path.intermediate_count + 1),
                "path": str(image_paths[index]) if index < len(image_paths) else None,
                "positions_direct": np.asarray(positions, dtype=float).tolist(),
            }
            for index, positions in enumerate(path.images)
        ],
        "species_by_atom": [str(symbol) for symbol in species_by_atom],
    }


class MigrationPathMixin:
    """The ``path`` stage: an evenly spaced chain between two structures."""

    def path(
        self,
        start_structure: str,
        end_structure: str,
        *,
        images: int = 3,
        match: bool = True,
        output_dir: str | None = None,
        cell_tolerance: float = 1e-6,
    ) -> CommandResult:
        """Write the migration-path images between two structures of one cell.

        ``images`` counts the intermediate images, so ``--images 3`` writes five
        directories, ``00`` to ``04``.  ``match`` pairs the atoms of the two
        endpoints by the assignment that makes the path shortest; switch it off
        when the two files are already written atom for atom in the same order
        and that order is the one to keep.
        """

        start_path = str(Path(start_structure).resolve())
        end_path = str(Path(end_structure).resolve())
        start = self.converter.read(start_path, canonicalize=False)
        end = self.converter.read(end_path, canonicalize=False)

        path = build_migration_path(
            start.lattice,
            start.species,
            start.counts,
            start.positions_direct,
            end.positions_direct,
            end_lattice=end.lattice,
            end_species=end.species,
            end_counts=end.counts,
            images=int(images),
            match=bool(match),
            cell_tolerance=float(cell_tolerance),
        )
        if path.path_length <= 1e-8:
            raise ValueError(
                "the two endpoints are the same structure, so there is no path between them"
            )

        degenerate = [
            index
            for index, positions in enumerate(path.images)
            if structure_errors(
                lattice=path.lattice,
                species=list(start.species),
                counts=list(start.counts),
                positions_direct=positions,
            )
        ]
        if degenerate:
            raise ValueError(
                f"image {degenerate[0]:02d} of the straight chain puts two atoms on one site, so "
                "the chain cannot be written: the two endpoints exchange atoms along a line that "
                "passes through them. Interpolate through an intermediate structure that steps "
                "around the collision instead"
            )

        run_id, run_dir = self.create_run_dir("path", label=f"{Path(start_path).stem}_to_{Path(end_path).stem}")
        destination = Path(output_dir).resolve() if output_dir else (run_dir / "path")
        destination.mkdir(parents=True, exist_ok=True)

        species_by_atom = [str(symbol) for symbol in expand_species(list(start.species), list(start.counts))]
        written: list[Path] = []
        for index, positions in enumerate(path.images):
            folder = destination / f"{index:02d}"
            folder.mkdir(parents=True, exist_ok=True)
            note = f"path image {index}/{path.intermediate_count + 1}"
            record = StructureRecord(
                comment=restage_comment(start.comment, f"{self.workflow_name} path", note),
                lattice=np.array(path.lattice, dtype=float, copy=True),
                species=list(start.species),
                counts=[int(value) for value in start.counts],
                positions_direct=np.array(positions, dtype=float, copy=True),
                positions_cartesian=np.asarray(positions, dtype=float) @ np.asarray(path.lattice, dtype=float),
                coordinate_mode="Direct",
                selective_dynamics=bool(start.selective_dynamics),
                selective_flags=None
                if start.selective_flags is None
                else [tuple(flags) for flags in start.selective_flags],
            )
            written.append(self.vasp_io.write(record, str(folder / "POSCAR"), wrap_positions=False))

        payload = _path_payload(path, species_by_atom, written)
        path_json = run_dir / "path.json"
        with path_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

        contacts = path.shortest_contacts()
        endpoint_contact = float(min(contacts[0], contacts[-1]))
        threshold = max(CONTACT_FLOOR, CONTACT_WARNING_FRACTION * endpoint_contact)
        warnings: list[str] = []
        pinched = [
            (index, float(value))
            for index, value in enumerate(contacts[1:-1], start=1)
            if float(value) < threshold
        ]
        if pinched:
            worst_index, worst = min(pinched, key=lambda item: item[1])
            warnings.append(
                f"image {worst_index:02d} brings two atoms to {worst:.2f} A, against {endpoint_contact:.2f} A "
                f"at the endpoints; a straight chain can push atoms through each other, so check the "
                f"images before running them, or interpolate through an intermediate structure"
            )
        reach = 0.5 * shortest_lattice_vector_length(np.asarray(path.lattice, dtype=float))
        if path.maximum_atom_displacement > reach:
            warnings.append(
                f"the longest single-atom step is {path.maximum_atom_displacement:.2f} A, more than half the "
                f"shortest lattice vector ({2.0 * reach:.2f} A); the chain takes the shortest periodic image, "
                f"which need not be the hop you meant"
            )
        if not path.matching.identity:
            reordered = int(np.count_nonzero(path.matching.partners != np.arange(len(path.matching.partners))))
            warnings.append(
                f"{reordered} atom(s) of the final structure were re-paired to make the path shortest; "
                f"the images are written in the atom order of the initial structure"
            )

        summary: dict[str, Any] = {
            "intermediate_images": int(path.intermediate_count),
            "images_written": len(written),
            "atoms": int(sum(int(value) for value in path.counts)),
            "moving_atoms": int(path.moved_atom_count),
            "path_length_ang": round(float(path.path_length), 5),
            "image_spacing_ang": round(float(path.image_spacing), 5),
            "maximum_atom_displacement_ang": round(float(path.maximum_atom_displacement), 5),
            "closest_contact_ang": round(float(np.min(contacts)), 5),
            "atoms_repaired": not bool(path.matching.identity),
            "matching_certificate_error": float(path.matching.certificate_error),
        }
        if warnings:
            summary["warnings"] = warnings
        artifacts = {
            "path_json": str(path_json.resolve()),
            "images": [str(item) for item in written],
        }
        manifest_path = self.write_manifest(
            stage="path",
            run_id=run_id,
            run_dir=run_dir,
            backend="native",
            inputs={"start_structure": start_path, "end_structure": end_path},
            parameters={
                "images": int(images),
                "match": bool(match),
                "output_dir": str(destination),
                "cell_tolerance": float(cell_tolerance),
            },
            artifacts=artifacts,
            summary=summary,
        )
        preview = format_path_table(path, written)
        moved = format_moved_atoms(path, species_by_atom)
        if moved:
            preview = f"{preview}\n\n{moved}"
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts=artifacts,
            summary=summary,
            payload={"path": payload, "path_preview": preview},
        )
