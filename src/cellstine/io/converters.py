"""Cross-format structure conversion helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.dependencies import DependencyManager
from ..core.validation import ensure_existing_file
from .models import StructureRecord
from .orientation import OrientationNormalizer
from .registry import FORMAT_EXTENSIONS
from .vasp import VaspIO


class StructureConverter:
    """Convert between VASP and selected other structure formats."""

    def __init__(
        self,
        *,
        dependency_manager: DependencyManager | None = None,
        vasp_io: VaspIO | None = None,
        normalizer: OrientationNormalizer | None = None,
    ) -> None:
        self.dependency_manager = dependency_manager or DependencyManager()
        self.vasp_io = vasp_io or VaspIO()
        self.normalizer = normalizer or OrientationNormalizer()

    def _detect_format(self, path: str | Path) -> str:
        suffix = Path(path).suffix.lower()
        for format_name, suffixes in FORMAT_EXTENSIONS.items():
            if suffix in suffixes:
                return format_name
        return suffix.lstrip(".")

    def read(self, path: str, *, canonicalize: bool = False, vacuum: float = 20.0) -> StructureRecord:
        source = ensure_existing_file(path)
        format_name = self._detect_format(source)
        if format_name == "vasp":
            record = self.vasp_io.read(str(source))
        elif format_name == "xyz":
            record = self._read_xyz(source, vacuum=vacuum)
        else:
            record = self._read_with_pymatgen(source)
        if canonicalize:
            record = self.normalizer.align_c_to_z(record)
        return record

    def write(self, record: StructureRecord, path: str, *, canonicalize: bool = False) -> Path:
        target = Path(path).resolve()
        format_name = self._detect_format(target)
        to_write = self.normalizer.align_c_to_z(record) if canonicalize else record
        if format_name == "vasp":
            return self.vasp_io.write(to_write, str(target), positions_are_cartesian=False, wrap_positions=False)
        if format_name == "xyz":
            return self._write_xyz(to_write, target)
        return self._write_with_pymatgen(to_write, target)

    def convert(self, source: str, target: str, *, canonicalize: bool = False, vacuum: float = 20.0) -> Path:
        record = self.read(source, canonicalize=canonicalize, vacuum=vacuum)
        return self.write(record, target, canonicalize=canonicalize)

    def _read_xyz(self, path: Path, *, vacuum: float) -> StructureRecord:
        with path.open("r", encoding="utf-8") as handle:
            lines = [line.rstrip("\n") for line in handle]
        if len(lines) < 2:
            raise ValueError(f"{path} is not a valid XYZ file")
        atom_count = int(lines[0].strip())
        comment = lines[1].strip() or f"Converted from {path.name}"
        atoms = []
        for line in lines[2 : 2 + atom_count]:
            tokens = line.split()
            atoms.append((tokens[0], [float(tokens[1]), float(tokens[2]), float(tokens[3])]))
        species_order: list[str] = []
        counts: list[int] = []
        for symbol, _ in atoms:
            if symbol not in species_order:
                species_order.append(symbol)
                counts.append(0)
            counts[species_order.index(symbol)] += 1
        cartesian = np.array([coords for _, coords in atoms], dtype=float)
        minima = cartesian.min(axis=0)
        maxima = cartesian.max(axis=0)
        span = np.maximum(maxima - minima, 1.0)
        box = np.diag(span + float(vacuum))
        shifted = cartesian - minima + 0.5 * float(vacuum)
        direct = shifted @ np.linalg.inv(box)
        return StructureRecord(
            comment=comment,
            lattice=box,
            species=species_order,
            counts=counts,
            positions_direct=direct,
            positions_cartesian=shifted,
            coordinate_mode="Cartesian",
            selective_dynamics=False,
            selective_flags=None,
            source_path=str(path),
            source_format="xyz",
            metadata={"vacuum_padding": float(vacuum)},
        )

    def _write_xyz(self, record: StructureRecord, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        species = []
        for symbol, count in zip(record.species, record.counts):
            species.extend([symbol] * int(count))
        with path.open("w", encoding="utf-8") as handle:
            handle.write(f"{record.natoms}\n")
            handle.write(f"{record.comment}\n")
            for symbol, position in zip(species, np.asarray(record.positions_cartesian, dtype=float)):
                handle.write(f"{symbol} {position[0]:.10f} {position[1]:.10f} {position[2]:.10f}\n")
        return path

    def _read_with_pymatgen(self, path: Path) -> StructureRecord:
        if not self.dependency_manager.has("pymatgen"):
            raise RuntimeError(f"reading '{path.suffix}' needs pymatgen; install cellstine[pymatgen]")
        from pymatgen.core import Molecule as PymatgenMolecule
        from pymatgen.core import Structure as PymatgenStructure

        try:
            structure = PymatgenStructure.from_file(str(path))
            ordered_species = [str(site.specie) for site in structure]
            species_out: list[str] = []
            counts_out: list[int] = []
            for symbol in ordered_species:
                if symbol not in species_out:
                    species_out.append(symbol)
                    counts_out.append(0)
                counts_out[species_out.index(symbol)] += 1
            return StructureRecord(
                comment=f"Converted from {path.name}",
                lattice=np.array(structure.lattice.matrix, dtype=float),
                species=species_out,
                counts=counts_out,
                positions_direct=np.array(structure.frac_coords, dtype=float),
                positions_cartesian=np.array(structure.cart_coords, dtype=float),
                coordinate_mode="Direct",
                selective_dynamics=False,
                selective_flags=None,
                source_path=str(path),
                source_format=path.suffix.lstrip(".").lower(),
                metadata={},
            )
        except Exception:
            molecule = PymatgenMolecule.from_file(str(path))
            tmp_xyz = path.with_suffix(".tmp.xyz")
            molecule.to(fmt="xyz", filename=str(tmp_xyz))
            try:
                return self._read_xyz(tmp_xyz, vacuum=20.0)
            finally:
                if tmp_xyz.exists():
                    tmp_xyz.unlink()

    def _write_with_pymatgen(self, record: StructureRecord, path: Path) -> Path:
        if not self.dependency_manager.has("pymatgen"):
            raise RuntimeError(f"writing '{path.suffix}' needs pymatgen; install cellstine[pymatgen]")
        from pymatgen.core import Structure as PymatgenStructure

        species = []
        for symbol, count in zip(record.species, record.counts):
            species.extend([symbol] * int(count))
        structure = PymatgenStructure(
            lattice=np.asarray(record.lattice, dtype=float),
            species=species,
            coords=np.asarray(record.positions_direct, dtype=float),
            coords_are_cartesian=False,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        structure.to(filename=str(path))
        return path
