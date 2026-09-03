"""Direct spglib-backed symmetry analysis and cell reduction."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..core.species import expand_species
from ..core.base import Base, run_output_suffix
from ..core.models import CommandResult
from ..io.converters import StructureConverter
from ..io.models import StructureRecord
from ..io.vasp import VaspIO
from ..core import bravais, reciprocal, symmetry3d
from ..core.idealisation import symmetrise_basis
from ..io import kpoints as kpoints_io
from .kpath_stage import BandPathMixin
from .models import EquivalentAtomGroup, SymmetryAnalysis, SymmetryOperation
from .records import (
    lattice_parameters as _lattice_parameters,
    record_from_atoms as _record_from_atoms,
    record_from_spglib_cell as _record_from_spglib_cell,
    species_type_map as _species_type_map,
)
from .reporting import format_symmetry_analysis
from .spglib_adapter import (
    crystal_system as _crystal_system,
    dataset_value as _dataset_value,
    has_inversion as _has_inversion,
)


def _write_analysis_file(path: Path, analysis: SymmetryAnalysis) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(analysis.to_dict(), handle, indent=2)
        handle.write("\n")
    return path.resolve()


class Symmetry(BandPathMixin, Base):
    """Analyse symmetry and reduce cells without routing through pymatgen."""

    workflow_name = "symmetry"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.converter = StructureConverter(dependency_manager=self.dependency_manager)
        self.vasp_io = VaspIO()

    def _spglib_cell(self, record: StructureRecord) -> tuple[tuple[np.ndarray, np.ndarray, list[int]], dict[int, str]]:
        species_by_atom = expand_species(record.species, record.counts)
        if len(species_by_atom) != int(record.natoms):
            raise ValueError("structure species/counts do not match atom positions")
        numbers, species_map = _species_type_map(species_by_atom)
        return (
            np.asarray(record.lattice, dtype=float),
            np.mod(np.asarray(record.positions_direct, dtype=float), 1.0),
            numbers,
        ), species_map

    def analyse_record(
        self,
        record: StructureRecord,
        *,
        structure_path: str | None = None,
        backend: str = "auto",
        symprec: float = 0.01,
        angle_tolerance: float = 5.0,
    ) -> SymmetryAnalysis:
        resolved_backend = self.dependency_manager.choose_symmetry_backend(backend, feature="symmetry analysis")
        if resolved_backend == "native":
            return self._native_analysis(record, structure_path=structure_path, symprec=float(symprec))
        return self._spglib_analysis(
            record,
            structure_path=structure_path,
            symprec=float(symprec),
            angle_tolerance=float(angle_tolerance),
        )

    def _native_analysis(self, record: StructureRecord, *, structure_path: str | None, symprec: float = 0.01) -> SymmetryAnalysis:
        """Return the symmetry of a cell from the native engine.

        The operations, their point group, the orbits of equivalent atoms and
        the centering translations are computed exactly (up to ``symprec``) by
        :mod:`cellstine.core.symmetry3d`.  Naming the space-group *type* and the
        Wyckoff letters needs the tabulated standard settings of all 230 groups
        and stays with the spglib backend.
        """

        lattice = np.asarray(record.lattice, dtype=float)
        species_by_atom = expand_species(record.species, record.counts)
        dataset = symmetry3d.analyse_symmetry(
            lattice,
            np.asarray(record.positions_direct, dtype=float),
            species_by_atom,
            symprec=float(symprec),
        )

        operations = [
            SymmetryOperation(
                rotation=np.asarray(rotation, dtype=int).tolist(),
                translation=tuple(float(value) for value in np.asarray(translation, dtype=float)),
            )
            for rotation, translation in zip(dataset.rotations, dataset.translations)
        ]

        grouped: dict[int, list[int]] = {}
        for atom_index, representative in enumerate(dataset.equivalent_atoms.tolist()):
            grouped.setdefault(int(representative), []).append(int(atom_index))
        groups = [
            EquivalentAtomGroup(
                group_id=f"atom_{group_index:03d}",
                species=str(species_by_atom[representative]),
                representative_index=int(representative) + 1,
                equivalent_indices=[int(index) + 1 for index in sorted(atom_indices)],
                multiplicity=int(len(atom_indices)),
                wyckoff=None,
            )
            for group_index, (representative, atom_indices) in enumerate(sorted(grouped.items()), start=1)
        ]

        notes = [
            "Native symmetry engine: exact integer lattice automorphisms plus a decorated-cell translation search.",
            f"Point group {dataset.point_group or 'unclassified'}; the lattice alone has point group {dataset.lattice_point_group or 'unclassified'}.",
            "Space-group type numbers, Hall symbols, and Wyckoff letters need the spglib backend.",
        ]
        if dataset.operation_count > 0 and not dataset.symmorphic_setting:
            notes.append("Some operations carry a non-lattice translation in this setting (screw axis or glide plane).")
        if len(dataset.primitive_translations) > 1:
            notes.append(
                f"The cell is {len(dataset.primitive_translations)}-fold non-primitive; "
                "`cellstine symmetry reduce --cell primitive` writes the primitive cell."
            )

        return SymmetryAnalysis(
            structure_path=structure_path,
            backend="native",
            atom_count=int(record.natoms),
            species=list(record.species),
            counts=[int(value) for value in record.counts],
            lattice_parameters=_lattice_parameters(lattice),
            point_group=dataset.point_group,
            crystal_system=dataset.crystal_system,
            lattice_type=symmetry3d.crystal_system_of_point_group(dataset.lattice_point_group),
            lattice_point_group=dataset.lattice_point_group,
            symmorphic_setting=bool(dataset.symmorphic_setting),
            centering_translation_count=int(len(dataset.primitive_translations)),
            laue=bool(dataset.has_inversion),
            operation_count=len(operations),
            operations=operations,
            equivalent_groups=groups,
            notes=notes,
        )

    def _spglib_analysis(
        self,
        record: StructureRecord,
        *,
        structure_path: str | None,
        symprec: float,
        angle_tolerance: float,
    ) -> SymmetryAnalysis:
        import spglib

        cell, species_map = self._spglib_cell(record)
        dataset = spglib.get_symmetry_dataset(cell, symprec=float(symprec), angle_tolerance=float(angle_tolerance))
        if dataset is None:
            raise RuntimeError("spglib could not determine symmetry for this structure")

        rotations = _dataset_value(dataset, "rotations")
        translations = _dataset_value(dataset, "translations")
        equivalent_atoms = _dataset_value(dataset, "equivalent_atoms")
        wyckoffs_raw = _dataset_value(dataset, "wyckoffs", [])
        wyckoffs = [str(value) for value in list(wyckoffs_raw)]
        species_by_atom = expand_species(record.species, record.counts)

        operations: list[SymmetryOperation] = []
        centering_translation_count = None
        symmorphic_setting = None
        if rotations is not None and translations is not None:
            rotation_array = np.asarray(rotations, dtype=int)
            translation_array = np.asarray(translations, dtype=float)
            centering_translations = symmetry3d.pure_translations(
                rotation_array,
                translation_array,
                symprec=float(symprec),
            )
            centering_translation_count = int(
                len(centering_translations)
            )
            symmorphic_setting = bool(
                symmetry3d._translations_are_centering(translation_array, centering_translations)
            )
            for rotation, translation in zip(rotations, translations):
                operations.append(
                    SymmetryOperation(
                        rotation=np.asarray(rotation, dtype=int).tolist(),
                        translation=tuple(float(value) for value in np.asarray(translation, dtype=float).tolist()),
                    )
                )

        groups: list[EquivalentAtomGroup] = []
        if equivalent_atoms is not None:
            grouped: dict[int, list[int]] = {}
            for atom_index, representative in enumerate(list(equivalent_atoms)):
                grouped.setdefault(int(representative), []).append(int(atom_index))
            for group_index, atom_indices in enumerate(grouped.values(), start=1):
                representative = int(atom_indices[0])
                groups.append(
                    EquivalentAtomGroup(
                        group_id=f"atom_{group_index:03d}",
                        species=str(species_by_atom[representative]),
                        representative_index=representative + 1,
                        equivalent_indices=[int(index) + 1 for index in sorted(atom_indices)],
                        multiplicity=int(len(atom_indices)),
                        wyckoff=wyckoffs[representative] if representative < len(wyckoffs) else None,
                    )
                )

        number_value = _dataset_value(dataset, "number")
        space_group_number = None if number_value is None else int(number_value)
        transformation = _dataset_value(dataset, "transformation_matrix")
        origin_shift = _dataset_value(dataset, "origin_shift")
        return SymmetryAnalysis(
            structure_path=structure_path,
            backend="spglib",
            atom_count=int(record.natoms),
            species=list(record.species),
            counts=[int(value) for value in record.counts],
            lattice_parameters=_lattice_parameters(np.asarray(record.lattice, dtype=float)),
            space_group_symbol=None if _dataset_value(dataset, "international") is None else str(_dataset_value(dataset, "international")),
            space_group_number=space_group_number,
            hall_symbol=None if _dataset_value(dataset, "hall") is None else str(_dataset_value(dataset, "hall")),
            point_group=None if _dataset_value(dataset, "pointgroup") is None else str(_dataset_value(dataset, "pointgroup")),
            crystal_system=_crystal_system(space_group_number),
            lattice_type=_crystal_system(space_group_number),
            laue=_has_inversion(None if rotations is None else np.asarray(rotations, dtype=int)),
            symmorphic_setting=symmorphic_setting,
            operation_count=len(operations),
            operations=operations,
            equivalent_groups=groups,
            wyckoffs=wyckoffs,
            centering_translation_count=centering_translation_count,
            transformation_matrix=None if transformation is None else np.asarray(transformation, dtype=float).tolist(),
            origin_shift=None if origin_shift is None else tuple(float(value) for value in np.asarray(origin_shift, dtype=float).tolist()),
            notes=["Exact crystallographic symmetry supplied by direct spglib backend."],
        )

    def _reduced_record(
        self,
        record: StructureRecord,
        *,
        cell_kind: str,
        symprec: float,
        angle_tolerance: float,
    ) -> StructureRecord:
        import spglib

        cell, species_map = self._spglib_cell(record)
        kind = str(cell_kind).lower()
        if kind == "primitive":
            reduced = spglib.find_primitive(cell, symprec=float(symprec), angle_tolerance=float(angle_tolerance))
        elif kind == "conventional":
            reduced = spglib.standardize_cell(
                cell,
                to_primitive=False,
                no_idealize=False,
                symprec=float(symprec),
                angle_tolerance=float(angle_tolerance),
            )
        elif kind == "refined":
            reduced = spglib.refine_cell(cell, symprec=float(symprec), angle_tolerance=float(angle_tolerance))
        else:
            raise ValueError("cell must be one of: primitive, conventional, refined")
        if reduced is None:
            raise RuntimeError(f"spglib could not produce a {kind} cell for this structure")
        return _record_from_spglib_cell(record, reduced, species_map, comment=f"{record.comment} | {kind} cell")

    def _native_primitive_record(self, record: StructureRecord, *, symprec: float) -> StructureRecord:
        """Return the primitive cell found by the native symmetry engine."""

        lattice, positions, species_by_atom = symmetry3d.primitive_cell(
            np.asarray(record.lattice, dtype=float),
            np.asarray(record.positions_direct, dtype=float),
            expand_species(record.species, record.counts),
            symprec=float(symprec),
        )
        return _record_from_atoms(record, lattice, positions, species_by_atom, comment=f"{record.comment} | primitive cell")

    def _native_conventional_record(
        self, record: StructureRecord, *, symprec: float, idealise: bool = False
    ) -> StructureRecord:
        """Return the conventional cell found by the native symmetry engine.

        The conventional cell is a property of the *translation lattice of the
        crystal*, not of the cell the file happens to be written in, so the
        primitive cell is taken first and classified with
        :func:`core.bravais.conventional_cell`.  That returns a basis spanning a
        superlattice cell of index ``multiplicity`` together with the fractional
        coordinates of the lattice points inside it, so the atoms of the
        conventional cell are exactly the primitive basis translated by each of
        those centring vectors: ``multiplicity`` times as many atoms in
        ``multiplicity`` times the volume, which the checks below assert.

        With ``idealise`` the conventional metric is additionally averaged over
        its own lattice point group
        (:func:`core.idealisation.symmetrise_basis`), so the cell obeys the
        symmetry of its Bravais class exactly rather than to within ``symprec``.
        Fractional coordinates are untouched by that step, so -- unlike spglib's
        ``refine_cell`` -- the atoms are not additionally snapped onto ideal
        Wyckoff positions; the deviation is recorded in the metadata so the size
        of the correction is visible.
        """

        lattice, positions, species_by_atom = symmetry3d.primitive_cell(
            np.asarray(record.lattice, dtype=float),
            np.asarray(record.positions_direct, dtype=float),
            expand_species(record.species, record.counts),
            symprec=float(symprec),
        )
        primitive = np.asarray(lattice, dtype=float)
        classification = bravais.conventional_cell(primitive)
        conventional_basis = np.asarray(classification.cell, dtype=float)
        multiplicity = int(classification.multiplicity)

        centrings = np.asarray(classification.centring_vectors, dtype=float).reshape(-1, 3)
        if centrings.shape[0] != multiplicity:
            raise RuntimeError("the conventional cell reported a centring count that is not its index")
        volume_ratio = abs(float(np.linalg.det(conventional_basis))) / abs(float(np.linalg.det(primitive)))
        if abs(volume_ratio - multiplicity) > 1e-6 * max(1.0, multiplicity):
            raise RuntimeError("the conventional cell volume is not the index times the primitive volume")

        conventional_coords = np.asarray(positions, dtype=float) @ np.asarray(
            classification.to_conventional, dtype=float
        )
        expanded = np.mod(conventional_coords[:, None, :] + centrings[None, :, :], 1.0).reshape(-1, 3)
        expanded_species = [symbol for symbol in species_by_atom for _ in range(multiplicity)]

        cell = conventional_basis
        deviation = 0.0
        if idealise:
            operations = symmetry3d.lattice_point_group(conventional_basis)
            idealised, deviation = symmetrise_basis(
                conventional_basis.T, operations, max_order=48, name="conventional cell"
            )
            cell = np.array(idealised.T, dtype=float)

        kind = "refined" if idealise else "conventional"
        conventional_record = _record_from_atoms(
            record,
            cell,
            expanded,
            expanded_species,
            comment=f"{record.comment} | {kind} cell ({classification.symbol})",
        )
        metadata = dict(conventional_record.metadata)
        metadata["bravais_symbol"] = classification.symbol
        metadata["centring"] = classification.centring
        metadata["crystal_system"] = classification.system
        metadata["conventional_multiplicity"] = multiplicity
        if idealise:
            metadata["metric_idealisation"] = float(deviation)
        return replace(conventional_record, metadata=metadata)

    def _lattice_reduced_record(
        self,
        record: StructureRecord,
        *,
        reduction: str,
        symprec: float,
        backend: str = "native",
    ) -> StructureRecord:
        method = str(reduction).lower()
        if method not in {"niggli", "delaunay"}:
            raise ValueError("reduction must be one of: niggli, delaunay")
        source_lattice = np.asarray(record.lattice, dtype=float)
        if str(backend) == "spglib":
            import spglib

            if method == "niggli":
                lattice = spglib.niggli_reduce(source_lattice, eps=float(symprec))
            else:
                lattice = spglib.delaunay_reduce(source_lattice, eps=float(symprec))
            if lattice is None:
                raise RuntimeError(f"spglib could not perform {method} lattice reduction")
        elif method == "niggli":
            lattice, _ = symmetry3d.niggli_reduce(source_lattice)
        else:
            lattice, _ = symmetry3d.delaunay_reduce(source_lattice)
        lattice_array = np.asarray(lattice, dtype=float)
        direct = np.mod(np.asarray(record.positions_cartesian, dtype=float) @ np.linalg.inv(lattice_array), 1.0)
        return StructureRecord(
            comment=f"{record.comment} | {method} lattice-reduced",
            lattice=lattice_array,
            species=list(record.species),
            counts=[int(value) for value in record.counts],
            positions_direct=direct,
            positions_cartesian=direct @ lattice_array,
            coordinate_mode="Direct",
            selective_dynamics=bool(record.selective_dynamics),
            selective_flags=None if record.selective_flags is None else [tuple(flags) for flags in record.selective_flags],
            source_path=record.source_path,
            source_format=record.source_format,
            metadata=dict(record.metadata),
        )

    def analyse(
        self,
        structure_path: str,
        *,
        backend: str = "auto",
        symprec: float = 0.01,
        angle_tolerance: float = 5.0,
    ) -> CommandResult:
        """Analyse crystallographic symmetry for a structure."""

        source = str(Path(structure_path).resolve())
        record = self.converter.read(source, canonicalize=False)
        analysis = self.analyse_record(
            record,
            structure_path=source,
            backend=backend,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
        run_id, run_dir = self.create_run_dir("analyse", label=Path(source).stem)
        analysis_json = _write_analysis_file(run_dir / "symmetry_analysis.json", analysis)
        summary = {
            "backend": analysis.backend,
            "space_group": analysis.space_group_symbol,
            "space_group_number": analysis.space_group_number,
            "point_group": analysis.point_group,
            "equivalent_groups": len(analysis.equivalent_groups),
            "operation_count": analysis.operation_count,
        }
        artifacts = {"analysis_json": str(analysis_json)}
        manifest_path = self.write_manifest(
            stage="analyse",
            run_id=run_id,
            run_dir=run_dir,
            backend=analysis.backend,
            inputs={"structure_path": source},
            parameters={"backend": backend, "symprec": float(symprec), "angle_tolerance": float(angle_tolerance)},
            artifacts=artifacts,
            summary=summary,
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts=artifacts,
            summary=summary,
            payload={"analysis": analysis.to_dict(), "symmetry_preview": self.format_analysis(analysis)},
        )

    def reduce(
        self,
        structure_path: str,
        *,
        cell: str = "primitive",
        backend: str = "auto",
        symprec: float = 0.01,
        angle_tolerance: float = 5.0,
        output_path: str | Path | None = None,
    ) -> CommandResult:
        """Write a primitive, conventional, or refined cell."""

        resolved_backend = self.dependency_manager.choose_symmetry_backend(backend, feature="cell reduction")
        kind = str(cell).lower()
        if kind not in {"primitive", "conventional", "refined"}:
            raise ValueError("cell must be one of: primitive, conventional, refined")
        source = str(Path(structure_path).resolve())
        record = self.converter.read(source, canonicalize=False)
        if resolved_backend == "spglib":
            reduced = self._reduced_record(record, cell_kind=kind, symprec=float(symprec), angle_tolerance=float(angle_tolerance))
        elif kind == "primitive":
            reduced = self._native_primitive_record(record, symprec=float(symprec))
        else:
            reduced = self._native_conventional_record(record, symprec=float(symprec), idealise=kind == "refined")
        run_id, run_dir = self.create_run_dir("reduce", label=Path(source).stem)
        destination = Path(output_path).resolve() if output_path is not None else self.output_root / f"symmetry_{cell}_{Path(source).stem}_{run_output_suffix(run_id).replace('_', '-')}.vasp"
        written = self.vasp_io.write(reduced, str(destination), positions_are_cartesian=False, wrap_positions=True)
        summary = {"backend": resolved_backend, "cell": kind, "atom_count": int(reduced.natoms)}
        artifacts = {"output_poscar": str(written)}
        manifest_path = self.write_manifest(
            stage="reduce",
            run_id=run_id,
            run_dir=run_dir,
            backend=resolved_backend,
            inputs={"structure_path": source},
            parameters={"cell": cell, "backend": backend, "symprec": float(symprec), "angle_tolerance": float(angle_tolerance)},
            artifacts=artifacts,
            summary=summary,
        )
        return self.result(manifest_path=manifest_path, run_dir=run_dir, artifacts=artifacts, summary=summary, payload={})

    def lattice_reduce(
        self,
        structure_path: str,
        *,
        reduction: str = "niggli",
        backend: str = "auto",
        symprec: float = 1e-5,
        output_path: str | Path | None = None,
    ) -> CommandResult:
        """Write a Niggli- or Delaunay-reduced lattice representation."""

        resolved_backend = self.dependency_manager.choose_symmetry_backend(backend, feature="lattice reduction")
        source = str(Path(structure_path).resolve())
        record = self.converter.read(source, canonicalize=False)
        reduced = self._lattice_reduced_record(
            record,
            reduction=reduction,
            symprec=float(symprec),
            backend=resolved_backend,
        )
        run_id, run_dir = self.create_run_dir("lattice_reduce", label=Path(source).stem)
        destination = Path(output_path).resolve() if output_path is not None else self.output_root / f"symmetry_{reduction}_{Path(source).stem}_{run_output_suffix(run_id).replace('_', '-')}.vasp"
        written = self.vasp_io.write(reduced, str(destination), positions_are_cartesian=False, wrap_positions=True)
        summary = {"backend": resolved_backend, "reduction": str(reduction).lower(), "atom_count": int(reduced.natoms)}
        artifacts = {"output_poscar": str(written)}
        manifest_path = self.write_manifest(
            stage="lattice-reduce",
            run_id=run_id,
            run_dir=run_dir,
            backend=resolved_backend,
            inputs={"structure_path": source},
            parameters={"reduction": reduction, "backend": backend, "symprec": float(symprec)},
            artifacts=artifacts,
            summary=summary,
        )
        return self.result(manifest_path=manifest_path, run_dir=run_dir, artifacts=artifacts, summary=summary, payload={})

    def kpoints(
        self,
        structure_path: str,
        *,
        spacing: float | None = None,
        divisions: Sequence[int] | None = None,
        mode: str = "gamma",
        shift: Sequence[float] | None = None,
        surface: bool = False,
        use_symmetry: bool = True,
        time_reversal: bool = True,
        explicit: bool | None = None,
        symprec: float = 0.01,
        output_path: str | Path | None = None,
    ) -> CommandResult:
        """Write a symmetry-reduced Brillouin-zone sampling mesh for a structure.

        Either ``spacing`` -- a largest allowed step in reciprocal space, in
        inverse angstrom and in the ``2 pi`` convention, the quantity VASP calls
        ``KSPACING`` -- or an explicit set of ``divisions`` fixes the mesh.
        ``surface`` pins the third division to one, which is what a slab with a
        vacuum gap needs: its bands do not disperse along the surface normal, so
        sampling that direction only multiplies the cost.

        The mesh is reduced by the rotation parts of the space-group operations
        of the cell itself, so a decorated cell keeps only the symmetry its atoms
        actually have, and by time reversal unless it is turned off.  The
        reported weights are exact orbit sizes and always add up to the size of
        the unreduced mesh.
        """

        source = str(Path(structure_path).resolve())
        record = self.converter.read(source, canonicalize=False)
        lattice = np.asarray(record.lattice, dtype=float)
        rotations = None
        centering_count = None
        if use_symmetry:
            rotations, operation_translations = symmetry3d.symmetry_operations(
                lattice,
                np.asarray(record.positions_direct, dtype=float),
                expand_species(record.species, record.counts),
                symprec=float(symprec),
            )
            centering_count = int(
                len(symmetry3d.pure_translations(rotations, operation_translations))
            )
        minimum = (1, 1, 1)
        if divisions is None and spacing is None:
            raise ValueError("give either a k-point spacing or explicit mesh divisions")
        counts = None if divisions is None else [int(value) for value in divisions]
        if counts is not None and surface:
            counts = [counts[0], counts[1], 1]
        if counts is None:
            counts = list(
                reciprocal.mesh_divisions_for_spacing(lattice, float(spacing), minimum=minimum)
            )
            if surface:
                counts[2] = 1
        mesh = reciprocal.build_mesh(
            lattice,
            divisions=counts,
            mode=str(mode),
            shift=shift,
            rotations=rotations,
            time_reversal=bool(time_reversal),
        )
        run_id, run_dir = self.create_run_dir("kpoints", label=Path(source).stem)
        destination = (
            Path(output_path).resolve()
            if output_path is not None
            else self.output_root
            / f"KPOINTS_{Path(source).stem}_{run_output_suffix(run_id).replace('_', '-')}"
        )
        comment = (
            f"{Path(source).stem} {counts[0]}x{counts[1]}x{counts[2]} "
            f"{'Gamma' if not any(mesh.shift) else 'shifted'} mesh"
        )
        written = kpoints_io.write_mesh(destination, mesh, explicit=explicit, comment=comment)
        summary = dict(mesh.summary())
        summary["structure"] = source
        summary["reduction_factor"] = float(mesh.full_point_count) / float(mesh.point_count)
        summary["points_per_zone_volume"] = reciprocal.kpoint_density(lattice, mesh.divisions)
        notes: list[str] = []
        if not mesh.symmetry_complete:
            notes.append(
                "the mesh is not invariant under every operation of this cell, so "
                f"only {mesh.operations_used} of {mesh.operations_given} could reduce it"
            )
        if centering_count is not None and centering_count > 1:
            notes.append(
                f"the cell is {centering_count}-fold non-primitive, so its zone is "
                f"{centering_count} times smaller than the zone of the primitive cell and every one "
                f"of these points carries {centering_count} folded points of that zone; "
                "`cellstine symmetry reduce --cell primitive` writes the primitive cell"
            )
        if notes:
            summary["note"] = "; ".join(notes)
        artifacts = {"kpoints": str(written)}
        manifest_path = self.write_manifest(
            stage="kpoints",
            run_id=run_id,
            run_dir=run_dir,
            backend="native",
            inputs={"structure_path": source},
            parameters={
                "spacing": None if spacing is None else float(spacing),
                "divisions": [int(value) for value in counts],
                "mode": str(mode),
                "shift": None if shift is None else [float(value) for value in shift],
                "surface": bool(surface),
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
                "points": mesh.points.tolist(),
                "weights": [int(value) for value in mesh.weights],
            },
        )

    @staticmethod
    def format_analysis(analysis: SymmetryAnalysis) -> str:
        """Return a compact CLI preview."""

        return format_symmetry_analysis(analysis)
