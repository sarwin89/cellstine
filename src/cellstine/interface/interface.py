"""Interface building and bulk-surface matching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from ..core.base import Base, run_output_suffix
from ..core.lattice import lattice_mismatch_fraction
from ..core.models import CommandResult
from ..core.transforms import strained_copy
from ..io.converters import StructureConverter
from ..io import native as io_mod
from ..io.orientation import OrientationNormalizer
from ..io.vasp import VaspIO


def parse_miller_notation(miller: str | Sequence[int]) -> tuple[int, int, int]:
    if not isinstance(miller, str):
        values = [int(value) for value in miller]
        if len(values) != 3:
            raise ValueError("Miller indices must have exactly three values")
        return values[0], values[1], values[2]
    raw = miller.strip()
    if "," not in raw and ";" not in raw and " " not in raw:
        tokens = []
        index = 0
        while index < len(raw):
            char = raw[index]
            if not char.isdigit():
                raise ValueError("compact Miller notation must look like 111, 001, 1x11, or 111x")
            token = char
            if index + 1 < len(raw) and raw[index + 1].lower() == "x":
                token += "x"
                index += 1
            tokens.append(token)
            index += 1
    else:
        tokens = [token.strip() for token in miller.replace(";", ",").replace(" ", ",").split(",") if token.strip()]
    if len(tokens) != 3:
        raise ValueError("Miller indices must be given as h,k,l such as 1,1,1 or compact notation such as 111, 001, or 111x")
    values = []
    for token in tokens:
        if token.lower().endswith("x"):
            values.append(-int(token[:-1]))
        else:
            values.append(int(token))
    if values == [0, 0, 0]:
        raise ValueError("Miller indices cannot all be zero")
    return int(values[0]), int(values[1]), int(values[2])


def _safe_token(value: object) -> str:
    text = str(value).strip().replace("-", "m").replace(".", "p")
    safe = [char if char.isalnum() or char in {"_", "m", "p"} else "_" for char in text]
    return "".join(safe).strip("_") or "x"


def _expand_species(record) -> list[str]:
    expanded = []
    for symbol, count in zip(record.species, record.counts):
        expanded.extend([str(symbol)] * int(count))
    return expanded


def _group_by_species(record, positions_direct: np.ndarray, selective_flags):
    expanded_species = _expand_species(record)
    grouped_positions: dict[str, list[np.ndarray]] = {}
    grouped_flags: dict[str, list[tuple[str, str, str] | None]] = {}
    species_order: list[str] = []
    for index, symbol in enumerate(expanded_species):
        if symbol not in grouped_positions:
            grouped_positions[symbol] = []
            grouped_flags[symbol] = []
            species_order.append(symbol)
        grouped_positions[symbol].append(np.array(positions_direct[index], dtype=float))
        if selective_flags is None:
            grouped_flags[symbol].append(None)
        else:
            grouped_flags[symbol].append(tuple(selective_flags[index]))
    ordered_positions = []
    ordered_flags: list[tuple[str, str, str]] = []
    counts = []
    for symbol in species_order:
        ordered_positions.extend(grouped_positions[symbol])
        counts.append(len(grouped_positions[symbol]))
        if selective_flags is not None:
            ordered_flags.extend(flag for flag in grouped_flags[symbol] if flag is not None)
    flags_out = ordered_flags if selective_flags is not None else None
    return np.array(ordered_positions, dtype=float), counts, species_order, flags_out


def _stack_structures(bottom, top, *, gap: float):
    bottom_cartesian = np.array(bottom.positions_cartesian, dtype=float, copy=True)
    top_cartesian = np.array(top.positions_cartesian, dtype=float, copy=True)
    lower_padding = 2.0
    bottom_min = float(bottom_cartesian[:, 2].min()) if bottom_cartesian.size else 0.0
    bottom_cartesian[:, 2] += lower_padding - bottom_min
    bottom_max = float(bottom_cartesian[:, 2].max()) if bottom_cartesian.size else 0.0
    top_min = float(top_cartesian[:, 2].min()) if top_cartesian.size else 0.0
    top_cartesian[:, 2] += bottom_max + float(gap) - top_min
    top_max = float(top_cartesian[:, 2].max()) if top_cartesian.size else bottom_max
    final_c_length = max(top_max + lower_padding, float(np.linalg.norm(bottom.lattice[2])))
    final_lattice = np.array(bottom.lattice, dtype=float, copy=True)
    final_lattice[2] = np.array([0.0, 0.0, final_c_length], dtype=float)

    bottom_direct = io_mod.cartesian_to_direct(bottom_cartesian, final_lattice)
    top_direct = io_mod.cartesian_to_direct(top_cartesian, final_lattice)
    bottom_positions, bottom_counts, bottom_species, bottom_flags = _group_by_species(bottom, bottom_direct, bottom.selective_flags)
    top_positions, top_counts, top_species, top_flags = _group_by_species(top, top_direct, top.selective_flags)

    positions = np.vstack((bottom_positions, top_positions))
    species = []
    counts = []
    grouped_flags = []
    grouped_positions = []
    all_species = bottom_species + [symbol for symbol in top_species if symbol not in bottom_species]
    for symbol in all_species:
        symbol_positions = []
        symbol_flags = []
        for current_species, current_counts, current_positions, current_flags in (
            (bottom_species, bottom_counts, bottom_positions, bottom_flags),
            (top_species, top_counts, top_positions, top_flags),
        ):
            offset = 0
            for species_name, count in zip(current_species, current_counts):
                block = current_positions[offset : offset + count]
                block_flags = None if current_flags is None else current_flags[offset : offset + count]
                if species_name == symbol:
                    symbol_positions.extend(np.array(item, dtype=float) for item in block)
                    if block_flags is not None:
                        symbol_flags.extend(tuple(item) for item in block_flags)
                offset += count
        if symbol_positions:
            species.append(symbol)
            counts.append(len(symbol_positions))
            grouped_positions.extend(symbol_positions)
            grouped_flags.extend(symbol_flags)
    flags_out = grouped_flags if grouped_flags else None
    return final_lattice, np.array(grouped_positions, dtype=float), counts, species, flags_out


class Interface(Base):
    """Top-level interface workflow."""

    workflow_name = "interface"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.converter = StructureConverter(dependency_manager=self.dependency_manager)
        self.normalizer = OrientationNormalizer()
        self.vasp_io = VaspIO()

    def _resolve_surface_input(
        self,
        *,
        path_or_manifest: str,
        kind: str,
        miller: str | Sequence[int] | None = None,
        layers: int = 4,
        vacuum: float = 15.0,
    ) -> tuple[str, dict[str, object]]:
        resolved_kind = str(kind).lower()
        candidate = Path(path_or_manifest).resolve()
        if candidate.name == "manifest.json":
            from ..core.manifests import RunManifest

            manifest = RunManifest.load(candidate)
            if "slab_poscar" in manifest.artifacts:
                return str(Path(str(manifest.artifacts["slab_poscar"])).resolve()), {"kind": "manifest"}
            raise ValueError(f"{candidate} does not contain a slab_poscar artifact")
        if resolved_kind in {"surface", "slab"}:
            return str(candidate), {"kind": resolved_kind}
        if resolved_kind != "bulk":
            raise ValueError("kind must be one of: bulk, slab, surface")
        from .surface import Surface

        slab_result = Surface(
            backend=self.backend,
            runs_root=self.runs_root,
            output_root=self.output_root,
            dependency_manager=self.dependency_manager,
        ).surface(
            bulk_poscar=path_or_manifest,
            miller=miller or "1,1,1",
            layers=int(layers),
            vacuum=float(vacuum),
        )
        return slab_result.artifacts["slab_poscar"], {"kind": "bulk", "surface_manifest": str(slab_result.manifest_path)}

    def build(
        self,
        *,
        bottom_input: str,
        top_input: str,
        bottom_kind: str = "surface",
        top_kind: str = "surface",
        bottom_miller: str | Sequence[int] | None = None,
        top_miller: str | Sequence[int] | None = None,
        bottom_layers: int = 4,
        top_layers: int = 4,
        bottom_vacuum: float = 15.0,
        top_vacuum: float = 15.0,
        gap: float = 3.0,
        output_path: str | None = None,
    ) -> CommandResult:
        backend = self.choose_backend(feature="interface.build")
        run_id, run_dir = self.create_run_dir("build", f"{Path(bottom_input).stem}_{Path(top_input).stem}")
        bottom_path, bottom_meta = self._resolve_surface_input(
            path_or_manifest=bottom_input,
            kind=bottom_kind,
            miller=bottom_miller,
            layers=bottom_layers,
            vacuum=bottom_vacuum,
        )
        top_path, top_meta = self._resolve_surface_input(
            path_or_manifest=top_input,
            kind=top_kind,
            miller=top_miller,
            layers=top_layers,
            vacuum=top_vacuum,
        )
        bottom = self.normalizer.align_c_to_z(self.converter.read(bottom_path))
        top = self.normalizer.align_c_to_z(self.converter.read(top_path))
        raw_mismatch = lattice_mismatch_fraction(bottom.lattice, top.lattice)
        strained_top = strained_copy(top, bottom.lattice)
        final_lattice, positions_direct, counts, species, flags = _stack_structures(bottom, strained_top, gap=float(gap))
        if output_path is None:
            output_suffix = run_output_suffix(run_id)
            destination = self.output_root / f"interface_gap{_safe_token(f'{float(gap):.2f}')}_{output_suffix}.vasp"
        else:
            destination = Path(output_path).resolve()
        output_record = bottom.copy()
        output_record.comment = f"{bottom.comment} | interface with {Path(top_path).stem}"
        output_record.lattice = final_lattice
        output_record.positions_direct = positions_direct
        output_record.positions_cartesian = io_mod.direct_to_cartesian(positions_direct, final_lattice)
        output_record.species = species
        output_record.counts = counts
        output_record.selective_flags = flags
        output_record.selective_dynamics = flags is not None
        self.vasp_io.write(output_record, str(destination), positions_are_cartesian=False, wrap_positions=False)
        manifest_path = self.write_manifest(
            stage="build",
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            inputs={
                "bottom_input": str(Path(bottom_input).resolve()),
                "top_input": str(Path(top_input).resolve()),
                "bottom_meta": bottom_meta,
                "top_meta": top_meta,
            },
            parameters={"gap": float(gap), "bottom_kind": str(bottom_kind), "top_kind": str(top_kind)},
            artifacts={"interface_poscar": destination},
            summary={"raw_inplane_mismatch": raw_mismatch, "total_atoms": int(sum(counts))},
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts={"interface_poscar": destination},
            summary={"raw_inplane_mismatch": raw_mismatch, "total_atoms": int(sum(counts))},
        )

    def match(
        self,
        *,
        bottom_bulk: str,
        top_bulk: str,
        bottom_millers: Sequence[str | Sequence[int]] | None = None,
        top_millers: Sequence[str | Sequence[int]] | None = None,
        bottom_layers_list: Sequence[int] | None = None,
        top_layers_list: Sequence[int] | None = None,
        vacuum: float = 15.0,
        max_strain: float = 0.05,
        output_path: str | None = None,
    ) -> CommandResult:
        backend = self.choose_backend(feature="interface.match")
        run_id, run_dir = self.create_run_dir("match", f"{Path(bottom_bulk).stem}_{Path(top_bulk).stem}")
        bottom_miller_values = list(bottom_millers or ["1,0,0", "1,1,0", "1,1,1"])
        top_miller_values = list(top_millers or ["1,0,0", "1,1,0", "1,1,1"])
        bottom_layer_values = [int(value) for value in (bottom_layers_list or [4])]
        top_layer_values = [int(value) for value in (top_layers_list or [4])]

        matches = []
        from .surface import Surface

        surface_tool = Surface(
            backend=self.backend,
            runs_root=self.runs_root,
            output_root=self.output_root,
            dependency_manager=self.dependency_manager,
        )
        for bottom_miller in bottom_miller_values:
            for bottom_layers in bottom_layer_values:
                bottom_surface = surface_tool.surface(
                    bulk_poscar=bottom_bulk,
                    miller=bottom_miller,
                    layers=bottom_layers,
                    vacuum=float(vacuum),
                    output_path=str(run_dir / f"bottom_{str(bottom_miller).replace(',', '')}_{bottom_layers}.vasp"),
                )
                bottom_record = self.normalizer.align_c_to_z(self.converter.read(bottom_surface.artifacts["slab_poscar"]))
                bottom_area = float(np.linalg.norm(np.cross(bottom_record.lattice[0], bottom_record.lattice[1])))
                for top_miller in top_miller_values:
                    for top_layers in top_layer_values:
                        top_surface = surface_tool.surface(
                            bulk_poscar=top_bulk,
                            miller=top_miller,
                            layers=top_layers,
                            vacuum=float(vacuum),
                            output_path=str(run_dir / f"top_{str(top_miller).replace(',', '')}_{top_layers}.vasp"),
                        )
                        top_record = self.normalizer.align_c_to_z(self.converter.read(top_surface.artifacts["slab_poscar"]))
                        strain = lattice_mismatch_fraction(bottom_record.lattice, top_record.lattice)
                        if strain > float(max_strain):
                            continue
                        matches.append(
                            {
                                "bottom_miller": parse_miller_notation(bottom_miller),
                                "top_miller": parse_miller_notation(top_miller),
                                "bottom_layers": int(bottom_layers),
                                "top_layers": int(top_layers),
                                "strain": float(strain),
                                "total_atoms": int(bottom_record.natoms + top_record.natoms),
                                "surface_area": float(bottom_area),
                                "bottom_slab": str(bottom_surface.artifacts["slab_poscar"]),
                                "top_slab": str(top_surface.artifacts["slab_poscar"]),
                            }
                        )
        matches.sort(key=lambda item: (item["strain"], item["total_atoms"], item["surface_area"]))
        results_path = Path(output_path).resolve() if output_path is not None else (run_dir / "matches.json")
        with results_path.open("w", encoding="utf-8") as handle:
            json.dump(matches, handle, indent=2)
        manifest_path = self.write_manifest(
            stage="match",
            run_id=run_id,
            run_dir=run_dir,
            backend=backend,
            inputs={"bottom_bulk": str(Path(bottom_bulk).resolve()), "top_bulk": str(Path(top_bulk).resolve())},
            parameters={
                "bottom_millers": [str(value) for value in bottom_miller_values],
                "top_millers": [str(value) for value in top_miller_values],
                "bottom_layers_list": bottom_layer_values,
                "top_layers_list": top_layer_values,
                "max_strain": float(max_strain),
            },
            artifacts={"matches_json": results_path},
            summary={"match_count": len(matches)},
        )
        return self.result(
            manifest_path=manifest_path,
            run_dir=run_dir,
            artifacts={"matches_json": results_path},
            summary={"match_count": len(matches)},
            payload={"best_match": matches[0] if matches else None},
        )
