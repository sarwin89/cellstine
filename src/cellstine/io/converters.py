"""Cross-format structure conversion helpers.

An XYZ file carries no cell, so :meth:`StructureConverter._read_xyz` builds one:
the bounding box of the molecule, each edge widened to at least one angstrom,
plus the requested vacuum, with the molecule shifted to sit centred in it.  That
padding is a guaranteed clearance, not merely a margin on the bounding box --
``aristotle-lean-reference/RequestProject/MolecularBox.lean`` proves in ``Cellstine.bboxShift_mem_cell``
that every atom lands strictly inside the box (so the direct coordinates written
out need no wrapping) and in ``Cellstine.bbox_image_separation`` that any two
atoms in different periodic images are at least the requested vacuum apart,
whatever the shape of the molecule.
"""

from __future__ import annotations

from itertools import chain, repeat
from pathlib import Path

import numpy as np

from ..core.dependencies import DependencyManager
from ..core.species import group_species
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
            record = self._read_with_pymatgen(source, vacuum=vacuum)
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
        padding = float(vacuum)
        if not np.isfinite(padding) or padding < 0.0:
            raise ValueError("XYZ vacuum padding must be finite and nonnegative")
        with path.open("r", encoding="utf-8") as handle:
            header = handle.readline()
            description = handle.readline()
            if not header or not description:
                raise ValueError(f"{path} is not a valid XYZ file")
            atom_count = int(header.strip())
            if atom_count < 1:
                raise ValueError(f"{path}: an XYZ molecular frame needs at least one atom")
            comment = description.strip() or f"Converted from {path.name}"
            atoms = []
            for index in range(atom_count):
                tokens = handle.readline().split()
                if len(tokens) < 4:
                    raise ValueError(f"{path}: XYZ atom {index + 1} is missing coordinates")
                coordinates = [float(value) for value in tokens[1:4]]
                if not all(np.isfinite(value) for value in coordinates):
                    raise ValueError(f"{path}: XYZ atom {index + 1} has a non-finite coordinate")
                atoms.append((tokens[0], coordinates))
        # A POSCAR groups its atoms by species, so the atoms are reordered here
        # rather than merely counted: an XYZ may list them in any order.
        species_order, counts, atom_order = group_species([symbol for symbol, _ in atoms])
        cartesian = np.array([coords for _, coords in atoms], dtype=float)[atom_order]
        minima = cartesian.min(axis=0)
        maxima = cartesian.max(axis=0)
        with np.errstate(over="ignore", invalid="ignore"):
            span = np.maximum(maxima - minima, 1.0)
            box = np.diag(span + padding)
            shifted = cartesian - minima + 0.5 * padding
        if not np.all(np.isfinite(box)) or not np.all(np.isfinite(shifted)):
            raise ValueError(f"{path}: XYZ molecular box exceeds the finite coordinate range")
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
            metadata={"vacuum_padding": padding},
        )

    def _write_xyz(self, record: StructureRecord, path: Path) -> Path:
        counts = [int(value) for value in record.counts]
        if len(record.species) != len(counts) or any(value < 0 for value in counts):
            raise ValueError("XYZ species and nonnegative counts must agree")
        atom_count = sum(counts)
        positions = np.asarray(record.positions_cartesian, dtype=float)
        if positions.shape != (atom_count, 3):
            raise ValueError("XYZ positions shape does not match atom counts")
        for start in range(0, atom_count, 8192):
            if not np.all(np.isfinite(positions[start : start + 8192])):
                raise ValueError("XYZ positions must be finite")
        comment = str(record.comment)
        if "\n" in comment or "\r" in comment:
            raise ValueError("an XYZ comment must occupy one line")
        species = chain.from_iterable(repeat(symbol, count) for symbol, count in zip(record.species, counts))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(f"{atom_count}\n")
            handle.write(f"{comment}\n")
            for symbol, position in zip(species, positions):
                handle.write(f"{symbol} {position[0]:.10f} {position[1]:.10f} {position[2]:.10f}\n")
        return path

    def _read_with_pymatgen(self, path: Path, *, vacuum: float = 20.0) -> StructureRecord:
        if not self.dependency_manager.has("pymatgen"):
            raise RuntimeError(f"reading '{path.suffix}' needs pymatgen; install cellstine[pymatgen]")
        from pymatgen.core import Molecule as PymatgenMolecule
        from pymatgen.core import Structure as PymatgenStructure

        try:
            structure = PymatgenStructure.from_file(str(path))
            species_out, counts_out, atom_order = group_species(
                [str(site.specie) for site in structure]
            )
            return StructureRecord(
                comment=f"Converted from {path.name}",
                lattice=np.array(structure.lattice.matrix, dtype=float),
                species=species_out,
                counts=counts_out,
                positions_direct=np.array(structure.frac_coords, dtype=float)[atom_order],
                positions_cartesian=np.array(structure.cart_coords, dtype=float)[atom_order],
                coordinate_mode="Direct",
                selective_dynamics=False,
                selective_flags=None,
                source_path=str(path),
                source_format=path.suffix.lstrip(".").lower(),
                metadata={},
            )
        except Exception:
            from tempfile import NamedTemporaryFile

            molecule = PymatgenMolecule.from_file(str(path))
            with NamedTemporaryFile(prefix="cellstine-molecule-", suffix=".xyz", delete=False) as handle:
                tmp_xyz = Path(handle.name)
            try:
                molecule.to(fmt="xyz", filename=str(tmp_xyz))
                record = self._read_xyz(tmp_xyz, vacuum=vacuum)
                if record.comment == f"Converted from {tmp_xyz.name}":
                    record.comment = f"Converted from {path.name}"
                record.source_path = str(path)
                record.source_format = path.suffix.lstrip(".").lower()
                return record
            finally:
                tmp_xyz.unlink(missing_ok=True)

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
