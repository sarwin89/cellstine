"""Adsorbate workflow wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..core.base import Base, run_output_suffix
from ..core.lattice import build_target_lattice
from ..core.models import CommandResult
from ..core.previews import format_bilayer_candidates
from ..interface.surface import Surface
from ..io.converters import StructureConverter
from ..io.vasp import VaspIO
from ..moire.find import run_find
from ..moire.molecule import place_molecule_on_site, transform_top_molecule


def _safe_token(value: object) -> str:
    text = str(value).strip().replace("-", "m").replace(".", "p")
    safe = [char if char.isalnum() or char in {"_", "m", "p"} else "_" for char in text]
    return "".join(safe).strip("_") or "x"


class Adsorbate(Base):
    """Molecule-on-substrate workflow."""

    workflow_name = "adsorbate"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.converter = StructureConverter(dependency_manager=self.dependency_manager)
        self.vasp_io = VaspIO()

    def _resolve_substrate(
        self,
        *,
        substrate_path: str,
        substrate_kind: str,
        miller: str | Sequence[int] | None = None,
        layers: int = 4,
        vacuum: float = 15.0,
        repeat_a: int = 1,
        repeat_b: int = 1,
        supercell_matrix: Sequence[int] | None = None,
    ) -> tuple[str, dict[str, object]]:
        resolved_kind = str(substrate_kind).lower()
        if resolved_kind in {"slab", "substrate", "patch", "surface"}:
            return str(Path(substrate_path).resolve()), {"substrate_kind": resolved_kind}
        if resolved_kind != "bulk":
            raise ValueError("substrate_kind must be one of: bulk, substrate, patch, surface, slab")
        slab_result = Surface(
            backend=self.backend,
            runs_root=self.runs_root,
            output_root=self.output_root,
            dependency_manager=self.dependency_manager,
        ).surface(
            bulk_poscar=substrate_path,
            miller=miller or "1,1,1",
            layers=int(layers),
            vacuum=float(vacuum),
            repeat_a=int(repeat_a),
            repeat_b=int(repeat_b),
            supercell_matrix=supercell_matrix,
        )
        return slab_result.artifacts["slab_poscar"], {"substrate_kind": "bulk", "surface_manifest": str(slab_result.manifest_path)}

    def place(
        self,
        *,
        substrate_poscar: str,
        molecule_poscar: str,
        substrate_kind: str = "substrate",
        miller: str | Sequence[int] | None = None,
        layers: int = 4,
        vacuum: float = 15.0,
        substrate_repeat_a: int = 1,
        substrate_repeat_b: int = 1,
        substrate_supercell_matrix: Sequence[int] | None = None,
        auto_repeat_substrate: bool = False,
        fit_padding: float = 0.15,
        site_type: str,
        site_index: int = 1,
        height: float = 2.5,
        rotation_deg: float = 0.0,
        surface_side: str = "top",
        output_path: str | None = None,
    ) -> CommandResult:
        backend = self.choose_backend(feature="adsorbate.place")
        run_id, run_dir = self.create_run_dir("place", f"{Path(substrate_poscar).stem}_{Path(molecule_poscar).stem}")
        output_suffix = run_output_suffix(run_id)
        molecule_path = Path(molecule_poscar).resolve()
        if molecule_path.suffix.lower() not in {".vasp", ".poscar", ".contcar", ""}:
            converted_molecule_path = run_dir / f"{_safe_token(molecule_path.stem)}_molecule.vasp"
            molecule_record = self.converter.read(str(molecule_path), canonicalize=False)
            self.vasp_io.write(molecule_record, str(converted_molecule_path), positions_are_cartesian=False, wrap_positions=False)
            resolved_molecule_path = converted_molecule_path
        else:
            resolved_molecule_path = molecule_path
        resolved_substrate, extra_inputs = self._resolve_substrate(
            substrate_path=substrate_poscar,
            substrate_kind=substrate_kind,
            miller=miller,
            layers=layers,
            vacuum=vacuum,
            repeat_a=substrate_repeat_a,
            repeat_b=substrate_repeat_b,
            supercell_matrix=substrate_supercell_matrix,
        )
        resolved_output_path = output_path or str(
            self.output_root
            / (
                f"adsorbate_{_safe_token(site_type)}{int(site_index):02d}_h{_safe_token(f'{float(height):.2f}')}"
                f"_rot{_safe_token(f'{float(rotation_deg):.2f}')}_{output_suffix}.vasp"
            )
        )
        run = place_molecule_on_site(
            substrate_poscar=resolved_substrate,
            molecule_poscar=str(resolved_molecule_path),
            site_type=str(site_type),
            site_index=int(site_index),
            height=float(height),
            rotation_deg=float(rotation_deg),
            surface_side=str(surface_side),
            auto_repeat_substrate=bool(auto_repeat_substrate),
            fit_padding=float(fit_padding),
            output_path=resolved_output_path,
        )
        manifest_path = self.write_manifest(
            stage="place",
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            inputs={
                "substrate_poscar": str(Path(substrate_poscar).resolve()),
                "molecule_poscar": str(molecule_path),
                "resolved_molecule_poscar": str(resolved_molecule_path),
                **extra_inputs,
            },
            parameters={
                "site_type": str(site_type),
                "site_index": int(site_index),
                "height": float(height),
                "rotation_deg": float(rotation_deg),
                "surface_side": str(surface_side),
                "substrate_repeat_a": int(substrate_repeat_a),
                "substrate_repeat_b": int(substrate_repeat_b),
                "substrate_supercell_matrix": list(substrate_supercell_matrix or []),
                "auto_repeat_substrate": bool(auto_repeat_substrate),
                "fit_padding": float(fit_padding),
            },
            artifacts={"output_poscar": run.output_path},
            summary={
                "site_type": run.site_type,
                "site_index": run.site_index,
                "molecule_atom_count": run.molecule_atom_count,
                "substrate_atom_count": run.substrate_atom_count,
            },
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts={"output_poscar": run.output_path},
            summary={"site_type": run.site_type, "site_index": run.site_index, "substrate_atom_count": run.substrate_atom_count},
            payload={"site_direct": run.site_direct, "site_cartesian": run.site_cartesian},
        )

    def move(
        self,
        *,
        poscar_path: str,
        target_cartesian: Sequence[float] | None = None,
        target_direct: Sequence[float] | None = None,
        rotation_deg: float = 0.0,
        z_cutoff: float | None = None,
        min_gap: float = 1.0,
        reframe_axes: str | Sequence[str] | None = "xy",
        output_path: str | None = None,
    ) -> CommandResult:
        backend = self.choose_backend(feature="adsorbate.move")
        run_id, run_dir = self.create_run_dir("move", Path(poscar_path).stem)
        output_suffix = run_output_suffix(run_id)
        target_token = "same"
        if target_cartesian is not None:
            target_token = "cart_" + "_".join(_safe_token(f"{float(value):.3f}") for value in target_cartesian)
        if target_direct is not None:
            target_token = "direct_" + "_".join(_safe_token(f"{float(value):.3f}") for value in target_direct)
        resolved_output_path = output_path or str(
            self.output_root
            / (
                f"move_{target_token}_rot{_safe_token(f'{float(rotation_deg):.2f}')}_{output_suffix}.vasp"
            )
        )
        run = transform_top_molecule(
            poscar_path=str(Path(poscar_path).resolve()),
            output_path=resolved_output_path,
            target_cartesian=target_cartesian,
            target_direct=target_direct,
            rotation_deg=float(rotation_deg),
            z_cutoff=z_cutoff,
            min_gap=float(min_gap),
            reframe_axes=reframe_axes,
        )
        manifest_path = self.write_manifest(
            stage="move",
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            inputs={"poscar_path": str(Path(poscar_path).resolve())},
            parameters={"target_cartesian": list(target_cartesian or []), "target_direct": list(target_direct or []), "rotation_deg": float(rotation_deg)},
            artifacts={"output_poscar": run.output_path},
            summary={"molecule_atom_count": run.molecule_atom_count, "substrate_atom_count": run.substrate_atom_count},
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts={"output_poscar": run.output_path},
            summary={"molecule_atom_count": run.molecule_atom_count},
            payload={"center_of_mass_after": run.center_of_mass_after},
        )

    def assemble(
        self,
        *,
        substrate_poscar: str,
        a_length: float,
        b_length: float | None = None,
        angle_deg: float = 60.0,
        nindex: int = 12,
        max_strain: float = 0.05,
        max_atoms: int | None = 2000,
        output_root: str | None = None,
        preview_limit: int = 10,
    ) -> CommandResult:
        backend = self.choose_backend(feature="adsorbate.assemble")
        run_id, run_dir = self.create_run_dir("assemble", Path(substrate_poscar).stem)
        substrate = self.converter.read(substrate_poscar)
        target_lattice = build_target_lattice(a_length=float(a_length), b_length=float(b_length or a_length), angle_deg=float(angle_deg))
        target_record = substrate.copy()
        target_record.comment = "Synthetic molecular assembly target lattice"
        target_record.lattice = target_lattice
        target_record.species = ["X"]
        target_record.counts = [1]
        target_record.positions_direct = target_record.positions_direct[:1].copy()
        target_record.positions_direct[0] = [0.0, 0.0, 0.0]
        target_record.positions_cartesian = target_record.positions_direct @ target_lattice
        target_path = run_dir / "assembly_target.vasp"
        self.vasp_io.write(target_record, str(target_path), positions_are_cartesian=False, wrap_positions=False)
        run = run_find(
            top_poscar=str(target_path),
            bottom_poscar=str(Path(substrate_poscar).resolve()),
            top_lattice=target_lattice,
            bottom_lattice=substrate.lattice,
            top_atoms=1,
            bottom_atoms=substrate.natoms,
            nindex=int(nindex),
            vector_tolerance=max_strain,
            vector_strain_tolerance=max_strain,
            candidate_tolerance=max_strain,
            strain_tolerance=max_strain,
            max_atoms=max_atoms,
            output_root=str(output_root or run_dir),
        )
        manifest_path = self.write_manifest(
            stage="assemble",
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            inputs={"substrate_poscar": str(Path(substrate_poscar).resolve())},
            parameters={"a_length": float(a_length), "b_length": float(b_length or a_length), "angle_deg": float(angle_deg), "max_strain": float(max_strain)},
            artifacts={"target_poscar": target_path, "results_dat": run.dat_path},
            summary={"candidate_count": len(run.candidates)},
        )
        preview = format_bilayer_candidates(run.candidates, limit=int(preview_limit)) if int(preview_limit) > 0 else ""
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts={"target_poscar": target_path, "results_dat": run.dat_path},
            summary={"candidate_count": len(run.candidates)},
            payload={"candidate_preview": preview},
        )
