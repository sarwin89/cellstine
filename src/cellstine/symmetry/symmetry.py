"""Direct spglib-backed symmetry analysis and cell reduction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..core.base import Base, run_output_suffix
from ..core.models import CommandResult
from ..io.converters import StructureConverter
from ..io.models import StructureRecord
from ..io.vasp import VaspIO
from ..core.lattice import infer_rotational_symmetry_angle, in_plane_lengths_and_angle


@dataclass
class SymmetryOperation:
    """One fractional symmetry operation."""

    rotation: list[list[int]]
    translation: tuple[float, float, float]


@dataclass
class EquivalentAtomGroup:
    """One set of symmetry-equivalent atom indices."""

    group_id: str
    species: str
    representative_index: int
    equivalent_indices: list[int]
    multiplicity: int
    wyckoff: str | None = None


@dataclass
class SymmetryAnalysis:
    """Serializable symmetry analysis result."""

    structure_path: str | None
    backend: str
    atom_count: int
    species: list[str]
    counts: list[int]
    lattice_parameters: dict[str, float]
    space_group_symbol: str | None = None
    space_group_number: int | None = None
    hall_symbol: str | None = None
    point_group: str | None = None
    crystal_system: str | None = None
    lattice_type: str | None = None
    laue: bool | None = None
    operation_count: int = 0
    operations: list[SymmetryOperation] = field(default_factory=list)
    equivalent_groups: list[EquivalentAtomGroup] = field(default_factory=list)
    wyckoffs: list[str] = field(default_factory=list)
    transformation_matrix: list[list[float]] | None = None
    origin_shift: tuple[float, float, float] | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "cellstine.symmetry_analysis.v1",
            "structure_path": self.structure_path,
            "backend": self.backend,
            "atom_count": int(self.atom_count),
            "species": list(self.species),
            "counts": [int(value) for value in self.counts],
            "lattice_parameters": dict(self.lattice_parameters),
            "space_group_symbol": self.space_group_symbol,
            "space_group_number": self.space_group_number,
            "hall_symbol": self.hall_symbol,
            "point_group": self.point_group,
            "crystal_system": self.crystal_system,
            "lattice_type": self.lattice_type,
            "laue": self.laue,
            "operation_count": int(self.operation_count),
            "operations": [asdict(operation) for operation in self.operations],
            "equivalent_groups": [asdict(group) for group in self.equivalent_groups],
            "wyckoffs": list(self.wyckoffs),
            "transformation_matrix": self.transformation_matrix,
            "origin_shift": None if self.origin_shift is None else [float(value) for value in self.origin_shift],
            "notes": list(self.notes),
        }


def _expanded_species(record: StructureRecord) -> list[str]:
    symbols: list[str] = []
    for symbol, count in zip(record.species, record.counts):
        symbols.extend([str(symbol)] * int(count))
    return symbols


def _species_type_map(species_by_atom: Sequence[str]) -> tuple[list[int], dict[int, str]]:
    order: list[str] = []
    numbers: list[int] = []
    mapping: dict[int, str] = {}
    for symbol in species_by_atom:
        if str(symbol) not in order:
            order.append(str(symbol))
            mapping[len(order)] = str(symbol)
        numbers.append(order.index(str(symbol)) + 1)
    return numbers, mapping


def _record_from_spglib_cell(
    source: StructureRecord,
    cell: tuple[Any, Any, Any],
    species_map: dict[int, str],
    *,
    comment: str,
) -> StructureRecord:
    lattice, positions, numbers = cell
    lattice_array = np.asarray(lattice, dtype=float)
    direct = np.mod(np.asarray(positions, dtype=float), 1.0)
    type_numbers = [int(value) for value in list(numbers)]
    atom_species = [species_map.get(number, f"X{number}") for number in type_numbers]

    ordered_species: list[str] = []
    for symbol in source.species:
        if symbol in atom_species and symbol not in ordered_species:
            ordered_species.append(str(symbol))
    for symbol in atom_species:
        if symbol not in ordered_species:
            ordered_species.append(str(symbol))

    grouped_positions: list[np.ndarray] = []
    counts: list[int] = []
    for symbol in ordered_species:
        indices = [index for index, atom_symbol in enumerate(atom_species) if atom_symbol == symbol]
        counts.append(len(indices))
        grouped_positions.extend(np.asarray(direct[index], dtype=float) for index in indices)

    output_direct = np.asarray(grouped_positions, dtype=float) if grouped_positions else np.zeros((0, 3), dtype=float)
    return StructureRecord(
        comment=comment,
        lattice=lattice_array,
        species=ordered_species,
        counts=counts,
        positions_direct=output_direct,
        positions_cartesian=output_direct @ lattice_array,
        coordinate_mode="Direct",
        selective_dynamics=False,
        selective_flags=None,
        source_path=source.source_path,
        source_format=source.source_format,
        metadata=dict(source.metadata),
    )


def _dataset_value(dataset: Any, key: str, default: Any = None) -> Any:
    if dataset is None:
        return default
    if hasattr(dataset, key):
        return getattr(dataset, key)
    try:
        return dataset[key]
    except Exception:
        return default


def _lattice_parameters(lattice: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(lattice, dtype=float)
    lengths = [float(np.linalg.norm(matrix[index])) for index in range(3)]
    angles = []
    for first, second in ((1, 2), (0, 2), (0, 1)):
        denominator = max(lengths[first] * lengths[second], 1e-12)
        cosine = np.clip(float(np.dot(matrix[first], matrix[second]) / denominator), -1.0, 1.0)
        angles.append(float(np.degrees(np.arccos(cosine))))
    return {
        "a": lengths[0],
        "b": lengths[1],
        "c": lengths[2],
        "alpha": angles[0],
        "beta": angles[1],
        "gamma": angles[2],
        "volume": abs(float(np.linalg.det(matrix))),
    }


def _crystal_system(number: int | None) -> str | None:
    if number is None:
        return None
    value = int(number)
    if 1 <= value <= 2:
        return "triclinic"
    if 3 <= value <= 15:
        return "monoclinic"
    if 16 <= value <= 74:
        return "orthorhombic"
    if 75 <= value <= 142:
        return "tetragonal"
    if 143 <= value <= 167:
        return "trigonal"
    if 168 <= value <= 194:
        return "hexagonal"
    if 195 <= value <= 230:
        return "cubic"
    return None


def _has_inversion(rotations: np.ndarray | None) -> bool | None:
    if rotations is None:
        return None
    inversion = -np.eye(3, dtype=int)
    return any(np.array_equal(np.asarray(rotation, dtype=int), inversion) for rotation in rotations)


def _write_analysis_file(path: Path, analysis: SymmetryAnalysis) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(analysis.to_dict(), handle, indent=2)
        handle.write("\n")
    return path.resolve()


class Symmetry(Base):
    """Analyse symmetry and reduce cells without routing through pymatgen."""

    workflow_name = "symmetry"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.converter = StructureConverter(dependency_manager=self.dependency_manager)
        self.vasp_io = VaspIO()

    def _spglib_cell(self, record: StructureRecord) -> tuple[tuple[np.ndarray, np.ndarray, list[int]], dict[int, str]]:
        species_by_atom = _expanded_species(record)
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
            return self._native_analysis(record, structure_path=structure_path)
        return self._spglib_analysis(
            record,
            structure_path=structure_path,
            symprec=float(symprec),
            angle_tolerance=float(angle_tolerance),
        )

    def _native_analysis(self, record: StructureRecord, *, structure_path: str | None) -> SymmetryAnalysis:
        a_length, b_length, gamma_deg = in_plane_lengths_and_angle(np.asarray(record.lattice, dtype=float))
        rotation_limit = infer_rotational_symmetry_angle(np.asarray(record.lattice, dtype=float))
        notes = [
            "Native symmetry analysis reports lattice geometry only.",
            "Install cellstine[symmetry] for exact space groups, primitive reduction, Wyckoff labels, and equivalent atom groups.",
        ]
        return SymmetryAnalysis(
            structure_path=structure_path,
            backend="native",
            atom_count=int(record.natoms),
            species=list(record.species),
            counts=[int(value) for value in record.counts],
            lattice_parameters=_lattice_parameters(np.asarray(record.lattice, dtype=float)),
            operation_count=0,
            notes=[
                *notes,
                f"Approximate in-plane rotational search limit: {int(rotation_limit)} degrees.",
                f"In-plane cell: a={a_length:.6g} A, b={b_length:.6g} A, gamma={gamma_deg:.6g} deg.",
            ],
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
        species_by_atom = _expanded_species(record)

        operations: list[SymmetryOperation] = []
        if rotations is not None and translations is not None:
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
            operation_count=len(operations),
            operations=operations,
            equivalent_groups=groups,
            wyckoffs=wyckoffs,
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

    def _lattice_reduced_record(self, record: StructureRecord, *, reduction: str, symprec: float) -> StructureRecord:
        import spglib

        method = str(reduction).lower()
        if method == "niggli":
            lattice = spglib.niggli_reduce(np.asarray(record.lattice, dtype=float), eps=float(symprec))
        elif method == "delaunay":
            lattice = spglib.delaunay_reduce(np.asarray(record.lattice, dtype=float), eps=float(symprec))
        else:
            raise ValueError("reduction must be one of: niggli, delaunay")
        if lattice is None:
            raise RuntimeError(f"spglib could not perform {method} lattice reduction")
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
        if resolved_backend != "spglib":
            raise RuntimeError("cell reduction requires spglib; install cellstine[symmetry]")
        source = str(Path(structure_path).resolve())
        record = self.converter.read(source, canonicalize=False)
        reduced = self._reduced_record(record, cell_kind=cell, symprec=float(symprec), angle_tolerance=float(angle_tolerance))
        run_id, run_dir = self.create_run_dir("reduce", label=Path(source).stem)
        destination = Path(output_path).resolve() if output_path is not None else self.output_root / f"symmetry_{cell}_{Path(source).stem}_{run_output_suffix(run_id).replace('_', '-')}.vasp"
        written = self.vasp_io.write(reduced, str(destination), positions_are_cartesian=False, wrap_positions=True)
        summary = {"backend": resolved_backend, "cell": str(cell).lower(), "atom_count": int(reduced.natoms)}
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
        if resolved_backend != "spglib":
            raise RuntimeError("lattice reduction requires spglib; install cellstine[symmetry]")
        source = str(Path(structure_path).resolve())
        record = self.converter.read(source, canonicalize=False)
        reduced = self._lattice_reduced_record(record, reduction=reduction, symprec=float(symprec))
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

    @staticmethod
    def format_analysis(analysis: SymmetryAnalysis) -> str:
        """Return a compact CLI preview."""

        lines = [f"Symmetry analysis ({analysis.backend})"]
        if analysis.space_group_symbol:
            lines.append(f"Space group: {analysis.space_group_symbol} ({analysis.space_group_number})")
        if analysis.point_group:
            lines.append(f"Point group: {analysis.point_group}")
        lines.append(f"Atoms: {analysis.atom_count}")
        lines.append(f"Operations: {analysis.operation_count}")
        if analysis.equivalent_groups:
            lines.append("Equivalent atom groups:")
            for group in analysis.equivalent_groups[:20]:
                represented = ",".join(str(index) for index in group.equivalent_indices)
                wyckoff = group.wyckoff or "-"
                lines.append(f"  {group.group_id} {group.species} mult={group.multiplicity} wyckoff={wyckoff} atoms={represented}")
            if len(analysis.equivalent_groups) > 20:
                lines.append(f"  ... {len(analysis.equivalent_groups) - 20} more group(s)")
        if analysis.notes:
            lines.append("Notes:")
            for note in analysis.notes[:4]:
                lines.append(f"- {note}")
        return "\n".join(lines)
