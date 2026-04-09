#!/usr/bin/env python3
"""Interactive user-facing CLI for CELLSTINE stages.

Made by Sarwin Chandran.
"""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import List, Sequence

APP_NAME = "CELLSTINE"
APP_EXPANSION = "CELL Superlattice Transformation INterface and Engine"
INPUT_DIR = Path("input")
RUNS_DIR = Path("runs")
OUTPUT_DIR = Path("output")


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """CLI help formatter with readable examples and visible defaults."""


@lru_cache(maxsize=1)
def _runtime_modules() -> SimpleNamespace:
    try:
        from moire import find as find_stage
        from moire import findn as findn_stage
        from moire import finder as finder_backend
        from moire import generator as generator_backend
        from moire import io as io_mod
        from moire import lattice as lattice_backend
        from moire import make as make_stage
        from moire import maken as maken_stage
        from moire import molecule as molecule_stage
        from moire import surface as surface_stage
        from moire import visualize as visualize_stage
    except ModuleNotFoundError as exc:
        if exc.name == "numpy":
            raise SystemExit(
                "CELLSTINE needs numpy installed to run searches and structure operations. "
                "Help is available, but this command needs the runtime dependencies."
            ) from exc
        raise
    return SimpleNamespace(
        find_stage=find_stage,
        findn_stage=findn_stage,
        finder_backend=finder_backend,
        generator_backend=generator_backend,
        io_mod=io_mod,
        lattice_backend=lattice_backend,
        make_stage=make_stage,
        maken_stage=maken_stage,
        molecule_stage=molecule_stage,
        surface_stage=surface_stage,
        visualize_stage=visualize_stage,
    )


def _latest_file(pattern: str, roots: Sequence[Path]) -> str | None:
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        candidates.extend(root.glob(pattern))
    ordered = sorted({path.resolve() for path in candidates}, key=lambda path: path.stat().st_mtime, reverse=True)
    if ordered:
        return str(ordered[0])
    return None


def _resolve_existing_path(raw: str, roots: Sequence[Path]) -> str:
    candidate = Path(raw).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    if candidate.is_absolute() or len(candidate.parts) > 1:
        return str(candidate)
    for root in roots:
        rooted = root / candidate
        if rooted.exists():
            return str(rooted.resolve())
    return str(candidate)


def _default_input_poscar(example_name: str = "mos2.vasp") -> str:
    latest = _latest_file("*.vasp", [INPUT_DIR])
    if latest is not None:
        return latest
    return str(INPUT_DIR / example_name)


def _parse_index_spec(raw: str) -> List[int]:
    values: List[int] = []
    for chunk in raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            step = 1 if end >= start else -1
            values.extend(list(range(start, end + step, step)))
        else:
            values.append(int(token))
    if not values:
        raise ValueError("please provide at least one index")
    ordered_unique = list(dict.fromkeys(values))
    return ordered_unique


def _suggest_find_defaults(top_lattice, bottom_lattice) -> dict[str, float | int | str]:
    runtime = _runtime_modules()
    top_a, top_b, _ = runtime.lattice_backend.in_plane_lengths_and_angle(top_lattice)
    bottom_a, bottom_b, _ = runtime.lattice_backend.in_plane_lengths_and_angle(bottom_lattice)
    top_scale = 0.5 * (top_a + top_b)
    bottom_scale = 0.5 * (bottom_a + bottom_b)
    primitive_mismatch = abs(top_scale - bottom_scale) / max(0.5 * (top_scale + bottom_scale), 1e-12)
    _, _, symmetry_lcm = runtime.lattice_backend.combined_symmetry_limit(top_lattice, bottom_lattice)

    if primitive_mismatch <= 2e-2:
        return {
            "min_angle": 0.0,
            "max_angle": float(symmetry_lcm),
            "angle_length_tolerance": 1e-5,
            "angle_strain_tolerance": 2e-3,
            "angle_merge_tolerance": 1e-3,
            "vector_tolerance": 2e-3,
            "vector_strain_tolerance": 2e-3,
            "candidate_tolerance": 2e-3,
            "strain_tolerance": 1e-2,
            "strain_layer": "avg",
            "max_atoms": 2000,
            "top_rows": 10,
            "output_root": "runs",
            "profile": "matched-lattice",
        }

    relaxed_strain = min(5e-2, max(1e-2, 0.5 * primitive_mismatch))
    return {
        "min_angle": 0.0,
        "max_angle": float(symmetry_lcm),
        "angle_length_tolerance": 1e-5,
        "angle_strain_tolerance": relaxed_strain,
        "angle_merge_tolerance": 1e-3,
        "vector_tolerance": 1e-2,
        "vector_strain_tolerance": relaxed_strain,
        "candidate_tolerance": 1e-2,
        "strain_tolerance": relaxed_strain,
        "strain_layer": "avg",
        "max_atoms": 2000,
        "top_rows": 10,
        "output_root": "runs",
        "profile": "mismatched-lattice",
    }


def _prompt_text(prompt_text: str, default_value: str | None = None, *, allow_empty: bool = False) -> str:
    suffix = f" [{default_value}]" if default_value not in {None, ""} else ""
    while True:
        user_text = input(f"{prompt_text}{suffix}: ").strip()
        if user_text:
            return user_text
        if default_value is not None:
            return default_value
        if allow_empty:
            return ""
        print("Please type a value and press Enter.")


def _prompt_int(prompt_text: str, default_value: int) -> int:
    while True:
        user_text = _prompt_text(prompt_text, str(default_value))
        try:
            return int(user_text)
        except ValueError:
            print("Please enter a whole number, for example 12.")


def _prompt_float(prompt_text: str, default_value: float) -> float:
    while True:
        user_text = _prompt_text(prompt_text, str(default_value))
        try:
            return float(user_text)
        except ValueError:
            print("Please enter a number, for example 0.01 or 3.35.")


def _prompt_optional_float(prompt_text: str, default_value: float | None = None) -> float | None:
    shown_default = "" if default_value is None else str(default_value)
    while True:
        user_text = _prompt_text(prompt_text, shown_default, allow_empty=True)
        if not user_text:
            return default_value
        try:
            return float(user_text)
        except ValueError:
            print("Please enter a number or leave it blank.")


def _prompt_yes_no(prompt_text: str, default_yes: bool = True) -> bool:
    default_label = "y" if default_yes else "n"
    while True:
        user_text = _prompt_text(prompt_text, default_label).strip().lower()
        if user_text in {"y", "yes"}:
            return True
        if user_text in {"n", "no"}:
            return False
        print("Please answer with y or n.")


def _prompt_choice(prompt_text: str, default_value: str, allowed: List[str]) -> str:
    allowed_map = {value.lower(): value for value in allowed}
    while True:
        user_text = _prompt_text(prompt_text, default_value).strip().lower()
        if user_text in allowed_map:
            return allowed_map[user_text]
        print("Please choose one of these options: " + ", ".join(allowed))


def _parse_angles(raw: str | None) -> List[float] | None:
    if not raw:
        return None
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def _parse_coordinate_vector(raw: str) -> List[float]:
    values = [float(token.strip()) for token in raw.split(",") if token.strip()]
    if len(values) not in {2, 3}:
        raise argparse.ArgumentTypeError(
            "coordinate vectors must have 2 values (x,y or u,v) or 3 values (x,y,z or u,v,w)"
        )
    return values


def _parse_matrix_values(raw: str) -> List[int]:
    tokens = [token.strip() for token in raw.replace(";", ",").split(",") if token.strip()]
    if len(tokens) != 4:
        raise argparse.ArgumentTypeError("please provide exactly four matrix values, for example 1,2,3,4")
    try:
        return [int(token) for token in tokens]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("matrix values must be whole numbers, for example 1,2,3,4") from exc


def _parse_miller(raw: str) -> List[int]:
    tokens = [token.strip() for token in raw.replace(";", ",").split(",") if token.strip()]
    if len(tokens) != 3:
        raise argparse.ArgumentTypeError("please provide Miller indices as h,k,l, for example 1,1,0")
    try:
        values = [int(token) for token in tokens]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Miller indices must be whole numbers, for example 1,1,0") from exc
    if values == [0, 0, 0]:
        raise argparse.ArgumentTypeError("Miller indices cannot all be zero")
    return values


def _expand_repeated_values(values: Sequence[object] | None, count: int, default_value: object | None = None) -> List[object | None]:
    if count <= 0:
        return []
    if values is None or len(values) == 0:
        return [default_value] * count
    resolved = list(values)
    if len(resolved) == 1 and count > 1:
        return resolved * count
    if len(resolved) != count:
        raise ValueError(f"expected either 1 value or {count} values, received {len(resolved)}")
    return resolved


def _format_fraction(value: float) -> str:
    return f"{value:.4f} ({100.0 * value:.2f}%)"


def _prompt_index_spec(prompt_text: str, default_value: str = "1") -> str:
    while True:
        user_text = _prompt_text(prompt_text, default_value)
        try:
            _parse_index_spec(user_text)
            return user_text
        except ValueError as exc:
            print(str(exc))


def _resolve_bottom_choice(poscar_a: str, poscar_b: str, bottom: str | None) -> tuple[str, str]:
    if Path(poscar_a).resolve() == Path(poscar_b).resolve():
        return poscar_a, poscar_b

    if bottom is None:
        bottom = _prompt_text(
            f"Which POSCAR should be the bottom layer? (a={poscar_a}, b={poscar_b})",
            "a",
        )

    key = bottom.strip().lower()
    if key in {"a", "first", "1"}:
        return poscar_a, poscar_b
    if key in {"b", "second", "2"}:
        return poscar_b, poscar_a
    if key == poscar_a.lower():
        return poscar_a, poscar_b
    if key == poscar_b.lower():
        return poscar_b, poscar_a
    raise ValueError("bottom choice must be a/first/1 or b/second/2")


def _latest_results_file() -> str | None:
    return _latest_file("*.dat", [RUNS_DIR])


def _latest_json_results_file() -> str | None:
    return _latest_file("*.json", [RUNS_DIR])


def _latest_poscar_file() -> str | None:
    return _latest_file("*.vasp", [OUTPUT_DIR, RUNS_DIR, INPUT_DIR, Path(".")])


def _suggest_make_defaults(records: List[dict]) -> dict[str, object]:
    return {
        "index_spec": "1",
        "interlayer": 3.35,
        "generator_tolerance": 1,
        "generator_tolerance_float": 1e-4,
        "output_dir": str(OUTPUT_DIR),
        "zfix": None,
        "record_count": len(records),
    }


def _print_find_summary(run: find_stage.FindRun, bottom_path: str, top_path: str, top_count: int) -> None:
    runtime = _runtime_modules()
    print(f"\n=== {APP_NAME} | Finding commensurate candidates (Made by Sarwin Chandran) ===")
    print(f"Run identifier      : {run.run_id}")
    print(f"Bottom layer        : {bottom_path}")
    print(f"Top layer           : {top_path}")
    print(f"Symmetry(top,bottom): ({run.symmetry_top}, {run.symmetry_bottom})")
    print(f"LCM angle limit     : {run.symmetry_lcm:.0f} deg")
    print(f"Search window       : {run.search_min_angle:.4f} -> {run.search_max_angle:.4f} deg")
    print(f"Angles checked      : {len(run.angle_values)}")
    print(f"Candidates found    : {len(run.candidates)}")
    print(f"Saved DAT           : {run.dat_path}")
    print("Units note          : angles in degrees; strain values are fractions (0.01 = 1%)")
    matrix_values = run.parameters.get("matrix_values")
    if matrix_values not in {None, ""}:
        print(
            "Matrix filter       : {values} | layer={layer} | mode={mode}".format(
                values=matrix_values,
                layer=run.parameters.get("matrix_layer", "either"),
                mode=run.parameters.get("matrix_match_mode", "absolute"),
            )
        )

    if run.candidates and top_count != 0:
        print("\nShown rows          : all" if top_count < 0 else f"\nShown rows          : {top_count}")
        print(runtime.finder_backend.format_results_table(run.candidates, limit=top_count))


def _print_findn_summary(run, bottom_path: str, upper_paths: Sequence[str], top_count: int) -> None:
    runtime = _runtime_modules()
    print(f"\n=== {APP_NAME} | Finding N-layer commensurate candidates (Made by Sarwin Chandran) ===")
    print(f"Run identifier      : {run.run_id}")
    print(f"Bottom layer        : {bottom_path}")
    for index, path in enumerate(upper_paths, start=1):
        print(f"Upper layer {index:>2d}     : {path}")
    for index, angle_values in enumerate(run.angle_values_by_layer, start=1):
        print(f"Angles checked U{index}   : {len(angle_values)}")
    print(f"Candidates found    : {len(run.candidates)}")
    print(f"Saved results       : {run.result_path}")
    print("Units note          : angles in degrees; strain values are fractions (0.01 = 1%)")
    if run.candidates and top_count != 0:
        print("\nShown rows          : all" if top_count < 0 else f"\nShown rows          : {top_count}")
        print(runtime.findn_stage.format_results_table(run.candidates, limit=top_count))


def _print_make_summary(run: make_stage.MakeRun) -> None:
    print(f"\n=== {APP_NAME} | Generating the commensurate superlattice (Made by Sarwin Chandran) ===")
    print(f"Selected index : {run.selected_index}")
    print(f"Angle (deg)    : {run.angle_deg:.4f}")
    print(f"Total atoms    : {run.total_atoms}")
    print(f"Output POSCAR  : {run.output_path}")


def _print_make_batch_summary(runs: List[make_stage.MakeRun]) -> None:
    print(f"\n=== {APP_NAME} | Generating the commensurate superlattice (Made by Sarwin Chandran) ===")
    print(f"Generated cells : {len(runs)}")
    for run in runs:
        print(
            "  idx {idx:3d} | angle {angle:8.4f} deg | atoms {atoms:6d} | {path}".format(
                idx=run.selected_index,
                angle=run.angle_deg,
                atoms=run.total_atoms,
                path=run.output_path,
            )
        )


def _print_maken_summary(run) -> None:
    print(f"\n=== {APP_NAME} | Generating the N-layer commensurate superlattice (Made by Sarwin Chandran) ===")
    print(f"Selected index : {run.selected_index}")
    print("Angles (deg)   : " + ", ".join(f"{angle:.4f}" for angle in run.angles_deg))
    print(f"Total atoms    : {run.total_atoms}")
    print(f"Output POSCAR  : {run.output_path}")


def _print_maken_batch_summary(runs: List) -> None:
    print(f"\n=== {APP_NAME} | Generating the N-layer commensurate superlattice (Made by Sarwin Chandran) ===")
    print(f"Generated cells : {len(runs)}")
    for run in runs:
        print(
            "  idx {idx:3d} | angles {angles} | atoms {atoms:6d} | {path}".format(
                idx=run.selected_index,
                angles=", ".join(f"{angle:.4f}" for angle in run.angles_deg),
                atoms=run.total_atoms,
                path=run.output_path,
            )
        )


def _print_molecule_summary(run: molecule_stage.MoleculeTransformRun) -> None:
    print(f"\n=== {APP_NAME} | Moving a top-side molecule (Made by Sarwin Chandran) ===")
    print(f"Molecule atoms        : {run.molecule_atom_count}")
    print(f"Substrate atoms       : {run.substrate_atom_count}")
    print(f"Detection z cutoff    : {run.z_cutoff:.6f} A")
    if run.gap_size == run.gap_size:
        print(f"Detected z gap        : {run.gap_size:.6f} A")
    print(
        "Center of mass before : ({0:.6f}, {1:.6f}, {2:.6f}) A".format(
            run.center_of_mass_before[0],
            run.center_of_mass_before[1],
            run.center_of_mass_before[2],
        )
    )
    print(
        "Center of mass after  : ({0:.6f}, {1:.6f}, {2:.6f}) A".format(
            run.center_of_mass_after[0],
            run.center_of_mass_after[1],
            run.center_of_mass_after[2],
        )
    )
    print(
        "Target point          : ({0:.6f}, {1:.6f}, {2:.6f}) A".format(
            run.target_cartesian[0],
            run.target_cartesian[1],
            run.target_cartesian[2],
        )
    )
    print(
        "Cell reframe shift    : ({0:.6f}, {1:.6f}, {2:.6f}) direct".format(
            run.reframe_shift_direct[0],
            run.reframe_shift_direct[1],
            run.reframe_shift_direct[2],
        )
    )
    print(f"Output POSCAR         : {run.output_path}")


def _print_layer_summary(run: molecule_stage.LayerShiftRun) -> None:
    print(f"\n=== {APP_NAME} | Shifting the upper layer (Made by Sarwin Chandran) ===")
    print(f"Upper-layer atoms     : {run.top_atom_count}")
    print(f"Lower-layer atoms     : {run.bottom_atom_count}")
    print(f"Detection z cutoff    : {run.z_cutoff:.6f} A")
    if run.gap_size == run.gap_size:
        print(f"Detected z gap        : {run.gap_size:.6f} A")
    print(
        "Applied shift         : ({0:.6f}, {1:.6f}, {2:.6f}) direct".format(
            run.shift_direct[0],
            run.shift_direct[1],
            run.shift_direct[2],
        )
    )
    print(
        "Applied shift         : ({0:.6f}, {1:.6f}, {2:.6f}) A".format(
            run.shift_cartesian[0],
            run.shift_cartesian[1],
            run.shift_cartesian[2],
        )
    )
    print(f"Output POSCAR         : {run.output_path}")


def _print_surface_summary(run) -> None:
    print(f"\n=== {APP_NAME} | Building a surface slab (Made by Sarwin Chandran) ===")
    print(f"Miller plane : ({run.miller[0]} {run.miller[1]} {run.miller[2]})")
    print(f"Layers       : {run.layers}")
    print(f"Vacuum (A)   : {run.vacuum:.3f}")
    print(f"Total atoms  : {run.total_atoms}")
    print(f"Output POSCAR: {run.output_path}")


def _print_visualize_summary(run) -> None:
    print(f"\n=== {APP_NAME} | Building the commensurate visualizer (Made by Sarwin Chandran) ===")
    print(f"Results type : {run.results_type}")
    print(f"Frames       : {run.frame_count}")
    print(f"Output HTML  : {run.output_path}")


def _run_find(args: argparse.Namespace) -> find_stage.FindRun:
    runtime = _runtime_modules()
    poscar_a = _resolve_existing_path(args.poscar_a, [INPUT_DIR, Path(".")])
    poscar_b = _resolve_existing_path(args.poscar_b, [INPUT_DIR, Path(".")])
    bottom_path, top_path = _resolve_bottom_choice(poscar_a, poscar_b, args.bottom)
    top_structure = runtime.io_mod.read_poscar(top_path)
    bottom_structure = runtime.io_mod.read_poscar(bottom_path)
    suggested = _suggest_find_defaults(top_structure.lattice, bottom_structure.lattice)

    min_angle = args.min_angle if args.min_angle is not None else float(suggested["min_angle"])
    max_angle = args.max_angle if args.max_angle is not None else float(suggested["max_angle"])
    angle_length_tolerance = (
        args.angle_length_tolerance
        if args.angle_length_tolerance is not None
        else float(suggested["angle_length_tolerance"])
    )
    angle_strain_tolerance = (
        args.angle_strain_tolerance
        if args.angle_strain_tolerance is not None
        else float(suggested["angle_strain_tolerance"])
    )
    angle_merge_tolerance = (
        args.angle_merge_tolerance
        if args.angle_merge_tolerance is not None
        else float(suggested["angle_merge_tolerance"])
    )
    vector_tolerance = args.vector_tolerance if args.vector_tolerance is not None else float(suggested["vector_tolerance"])
    vector_strain_tolerance = (
        args.vector_strain_tolerance
        if args.vector_strain_tolerance is not None
        else float(suggested["vector_strain_tolerance"])
    )
    candidate_tolerance = (
        args.candidate_tolerance
        if args.candidate_tolerance is not None
        else float(suggested["candidate_tolerance"])
    )
    strain_tolerance = (
        args.strain_tolerance
        if args.strain_tolerance is not None
        else float(suggested["strain_tolerance"])
    )
    max_atoms = args.max_atoms if args.max_atoms is not None else int(suggested["max_atoms"])

    matrix_values = (
        _parse_matrix_values(args.matrix_values)
        if isinstance(args.matrix_values, str)
        else args.matrix_values
    )

    run = runtime.find_stage.run_find(
        top_poscar=top_path,
        bottom_poscar=bottom_path,
        top_lattice=top_structure.lattice,
        bottom_lattice=bottom_structure.lattice,
        top_atoms=top_structure.natoms * int(args.top_c_repeat),
        bottom_atoms=bottom_structure.natoms * int(args.bottom_c_repeat),
        nindex=args.nindex,
        min_angle=min_angle,
        max_angle=max_angle,
        angle_step=args.angle_step,
        explicit_angles=_parse_angles(args.angles),
        angle_length_tolerance=angle_length_tolerance,
        angle_strain_tolerance=angle_strain_tolerance,
        angle_merge_tolerance=angle_merge_tolerance,
        vector_tolerance=vector_tolerance,
        vector_strain_tolerance=vector_strain_tolerance,
        candidate_tolerance=candidate_tolerance,
        strain_tolerance=strain_tolerance,
        strain_layer=args.strain_layer,
        min_atoms=args.min_atoms,
        max_atoms=max_atoms,
        dedupe=not args.no_dedupe,
        unique_strain_tolerance=args.unique_strain_tolerance,
        unique_ratio_tolerance=args.unique_ratio_tolerance,
        matrix_values=matrix_values,
        matrix_layer=args.matrix_layer,
        matrix_match_mode=args.matrix_match_mode,
        output_root=args.output_root,
        top_c_repeat=args.top_c_repeat,
        bottom_c_repeat=args.bottom_c_repeat,
        workers=args.workers,
    )

    _print_find_summary(run, bottom_path, top_path, args.top)
    return run


def _run_findn(args: argparse.Namespace):
    runtime = _runtime_modules()
    bottom_path = _resolve_existing_path(args.bottom_poscar, [INPUT_DIR, Path(".")])
    upper_paths = [_resolve_existing_path(path, [INPUT_DIR, Path(".")]) for path in args.upper_poscars]
    bottom_structure = runtime.io_mod.read_poscar(bottom_path)
    upper_structures = [runtime.io_mod.read_poscar(path) for path in upper_paths]

    upper_count = len(upper_paths)
    min_angle_values = _expand_repeated_values(args.min_angles, upper_count, None)
    max_angle_values = _expand_repeated_values(args.max_angles, upper_count, None)
    explicit_angle_values = _expand_repeated_values(args.angles, upper_count, None)
    upper_c_repeats = [int(value) for value in _expand_repeated_values(args.upper_c_repeat, upper_count, 1)]

    suggested_profiles = [_suggest_find_defaults(structure.lattice, bottom_structure.lattice) for structure in upper_structures]
    min_angles = [
        float(value if value is not None else suggested_profiles[index]["min_angle"])
        for index, value in enumerate(min_angle_values)
    ]
    max_angles = [
        float(value if value is not None else suggested_profiles[index]["max_angle"])
        for index, value in enumerate(max_angle_values)
    ]
    explicit_angles_by_layer = [_parse_angles(value) if isinstance(value, str) and value else None for value in explicit_angle_values]
    angle_length_tolerance = (
        args.angle_length_tolerance
        if args.angle_length_tolerance is not None
        else max(float(profile["angle_length_tolerance"]) for profile in suggested_profiles)
    )
    angle_strain_tolerance = (
        args.angle_strain_tolerance
        if args.angle_strain_tolerance is not None
        else max(float(profile["angle_strain_tolerance"]) for profile in suggested_profiles)
    )
    angle_merge_tolerance = (
        args.angle_merge_tolerance
        if args.angle_merge_tolerance is not None
        else max(float(profile["angle_merge_tolerance"]) for profile in suggested_profiles)
    )
    vector_tolerance = (
        args.vector_tolerance
        if args.vector_tolerance is not None
        else max(float(profile["vector_tolerance"]) for profile in suggested_profiles)
    )
    vector_strain_tolerance = (
        args.vector_strain_tolerance
        if args.vector_strain_tolerance is not None
        else max(float(profile["vector_strain_tolerance"]) for profile in suggested_profiles)
    )
    candidate_tolerance = (
        args.candidate_tolerance
        if args.candidate_tolerance is not None
        else max(float(profile["candidate_tolerance"]) for profile in suggested_profiles)
    )
    pair_strain_tolerance = (
        args.pair_strain_tolerance
        if args.pair_strain_tolerance is not None
        else max(float(profile["strain_tolerance"]) for profile in suggested_profiles)
    )
    max_atoms = (
        args.max_atoms
        if args.max_atoms is not None
        else int(max(int(profile["max_atoms"]) for profile in suggested_profiles))
    )

    run = runtime.findn_stage.run_findn(
        bottom_poscar=bottom_path,
        upper_poscars=upper_paths,
        bottom_lattice=bottom_structure.lattice,
        upper_lattices=[structure.lattice for structure in upper_structures],
        bottom_atoms=bottom_structure.natoms * int(args.bottom_c_repeat),
        upper_atoms=[structure.natoms * repeat for structure, repeat in zip(upper_structures, upper_c_repeats)],
        nindex=args.nindex,
        min_angles=min_angles,
        max_angles=max_angles,
        angle_step=args.angle_step,
        explicit_angles_by_layer=explicit_angles_by_layer,
        angle_length_tolerance=angle_length_tolerance,
        angle_strain_tolerance=angle_strain_tolerance,
        angle_merge_tolerance=angle_merge_tolerance,
        vector_tolerance=vector_tolerance,
        vector_strain_tolerance=vector_strain_tolerance,
        candidate_tolerance=candidate_tolerance,
        pair_strain_tolerance=pair_strain_tolerance,
        max_atoms=max_atoms,
        dedupe=not args.no_dedupe,
        unique_strain_tolerance=args.unique_strain_tolerance,
        unique_ratio_tolerance=args.unique_ratio_tolerance,
        output_root=args.output_root,
        bottom_c_repeat=args.bottom_c_repeat,
        upper_c_repeats=upper_c_repeats,
        workers=args.workers,
    )
    _print_findn_summary(run, bottom_path, upper_paths, args.top)
    return run


def _run_make(args: argparse.Namespace) -> List[make_stage.MakeRun]:
    runtime = _runtime_modules()
    results_path = _resolve_existing_path(args.results, [RUNS_DIR, Path(".")])
    _, _, records, _ = runtime.generator_backend.parse_results(results_path)
    if not records:
        raise ValueError("results file has no candidate records")
    if args.index is None:
        raise ValueError("please choose one or more candidate rows with --index, for example --index 1 or --index 1,2,5-7")

    indexes = _parse_index_spec(args.index)
    interlayer = 3.35 if args.interlayer is None else args.interlayer
    if len(indexes) > 1 and args.output is not None:
        raise ValueError("use --output only with a single index; use --output-dir for multiple indexes")

    runs = runtime.make_stage.generate_many_from_results(
        results_path,
        indexes=indexes,
        interlayer_distance=interlayer,
        output_path=args.output if len(indexes) == 1 else None,
        output_dir=args.output_dir,
        tolerance=args.generator_tolerance,
        tolerance_float=args.generator_tolerance_float,
        zfix=args.zfix,
        top_c_repeat=args.top_c_repeat,
        bottom_c_repeat=args.bottom_c_repeat,
        workers=args.workers,
    )
    if len(runs) == 1:
        _print_make_summary(runs[0])
    else:
        _print_make_batch_summary(runs)
    return runs


def _run_maken(args: argparse.Namespace):
    runtime = _runtime_modules()
    results_path = _resolve_existing_path(args.results, [RUNS_DIR, Path(".")])
    meta, candidates = runtime.findn_stage.parse_results(results_path)
    if not candidates:
        raise ValueError("N-layer results file has no candidate records")
    if args.index is None:
        raise ValueError("please choose one or more candidate rows with --index, for example --index 1 or --index 1,2,5-7")

    indexes = _parse_index_spec(args.index)
    upper_count = len(meta.get("upper_poscars", []))
    interlayers = [float(value) for value in _expand_repeated_values(args.interlayers, upper_count, 3.35)]
    upper_repeat_values = _expand_repeated_values(args.upper_c_repeat, upper_count, None)
    upper_c_repeats = None
    if any(value is not None for value in upper_repeat_values):
        upper_c_repeats = [int(value if value is not None else 1) for value in upper_repeat_values]
    runs = runtime.maken_stage.generate_many_from_results(
        results_path,
        indexes=indexes,
        interlayers=interlayers,
        output_dir=args.output_dir,
        bottom_c_repeat=args.bottom_c_repeat,
        upper_c_repeats=upper_c_repeats,
        zfix=args.zfix,
    )
    if len(runs) == 1:
        _print_maken_summary(runs[0])
    else:
        _print_maken_batch_summary(runs)
    return runs


def _run_molecule(args: argparse.Namespace) -> molecule_stage.MoleculeTransformRun:
    runtime = _runtime_modules()
    run = runtime.molecule_stage.transform_top_molecule(
        _resolve_existing_path(args.poscar, [OUTPUT_DIR, RUNS_DIR, INPUT_DIR, Path(".")]),
        output_path=args.output,
        target_cartesian=args.target_cart,
        target_direct=args.target_direct,
        rotation_deg=args.rotate,
        z_cutoff=args.z_cutoff,
        min_gap=args.min_gap,
        reframe_axes=args.reframe,
    )
    _print_molecule_summary(run)
    return run


def _run_layer(args: argparse.Namespace) -> molecule_stage.LayerShiftRun:
    runtime = _runtime_modules()
    run = runtime.molecule_stage.shift_top_layer(
        _resolve_existing_path(args.poscar, [OUTPUT_DIR, RUNS_DIR, INPUT_DIR, Path(".")]),
        output_path=args.output,
        shift_cartesian=args.shift_cart,
        shift_direct=args.shift_direct,
        z_cutoff=args.z_cutoff,
        min_gap=args.min_gap,
    )
    _print_layer_summary(run)
    return run


def _run_surface(args: argparse.Namespace):
    runtime = _runtime_modules()
    run = runtime.surface_stage.build_surface(
        _resolve_existing_path(args.bulk_poscar, [INPUT_DIR, Path(".")]),
        miller=tuple(_parse_miller(args.miller) if isinstance(args.miller, str) else args.miller),
        layers=args.layers,
        vacuum=args.vacuum,
        repeat_a=args.repeat_a,
        repeat_b=args.repeat_b,
        output_path=args.output,
    )
    _print_surface_summary(run)
    return run


def _run_visualize(args: argparse.Namespace):
    runtime = _runtime_modules()
    indices = _parse_index_spec(args.index) if args.index is not None else None
    run = runtime.visualize_stage.build_visualization(
        _resolve_existing_path(args.results, [RUNS_DIR, Path(".")]),
        indices=indices,
        output_path=args.output,
        interlayer=args.interlayer,
        interlayer_bottom_middle=args.interlayer_bottom_middle,
        interlayer_middle_top=args.interlayer_middle_top,
        top_c_repeat=args.top_c_repeat,
        bottom_c_repeat=args.bottom_c_repeat,
        middle_c_repeat=args.middle_c_repeat,
    )
    _print_visualize_summary(run)
    return run


def _prompt_make_configuration(results_path: str, records: List[dict]) -> argparse.Namespace:
    suggested_make = _suggest_make_defaults(records)
    print("\nGenerating the commensurate superlattice:")
    print(f"- available candidate indices: 1-{len(records)}")
    print("- choose a single row like 1 or several rows like 1,2,5-7")
    print(f"- interlayer gap is in angstrom; a common starting value is {float(suggested_make['interlayer']):.2f}")
    print(f"- generated POSCARs will go to: {suggested_make['output_dir']}")

    index_spec = _prompt_index_spec("Which candidate rows would you like to generate?", str(suggested_make["index_spec"]))
    interlayer = _prompt_float("Interlayer gap in angstrom", float(suggested_make["interlayer"]))
    output_dir = _prompt_text("Folder for generated POSCARs", str(suggested_make["output_dir"]))

    generator_tolerance = int(suggested_make["generator_tolerance"])
    generator_tolerance_float = float(suggested_make["generator_tolerance_float"])
    zfix = None
    if _prompt_yes_no("Adjust advanced generator settings?", False):
        generator_tolerance = _prompt_int("Integer padding tolerance for the generator", generator_tolerance)
        generator_tolerance_float = _prompt_float("Floating-point tolerance for the generator", generator_tolerance_float)
        zfix = _prompt_optional_float("Optional zfix cutoff in angstrom (leave blank to skip)", None)

    return argparse.Namespace(
        results=results_path,
        index=index_spec,
        interlayer=interlayer,
        output=None,
        output_dir=output_dir,
        generator_tolerance=generator_tolerance,
        generator_tolerance_float=generator_tolerance_float,
        zfix=zfix,
    )


def _interactive_make_from_results(results_path: str) -> None:
    runtime = _runtime_modules()
    resolved_results_path = _resolve_existing_path(results_path, [RUNS_DIR, Path(".")])
    _, _, records, _ = runtime.generator_backend.parse_results(resolved_results_path)
    print(f"Loaded {len(records)} candidate row(s) from {resolved_results_path}")
    make_args = _prompt_make_configuration(resolved_results_path, records)
    _run_make(make_args)


def _interactive_move_top_group() -> None:
    runtime = _runtime_modules()
    poscar_path = _prompt_text("Path to stacked POSCAR", _latest_poscar_file() or str(OUTPUT_DIR / "stacked.vasp"))
    poscar_path = _resolve_existing_path(poscar_path, [OUTPUT_DIR, RUNS_DIR, INPUT_DIR, Path(".")])
    structure = runtime.io_mod.read_poscar(poscar_path)

    mode = _prompt_choice("What would you like to adjust on the top side? molecule / layer", "molecule", ["molecule", "layer"])
    coordinate_mode = _prompt_choice("Which coordinate system would you like to use? direct / cartesian", "direct", ["direct", "cartesian"])
    z_cutoff = _prompt_optional_float("Optional manual z cutoff in angstrom (leave blank to auto-detect)", None)
    min_gap = 1.0 if z_cutoff is None else 0.0
    if z_cutoff is None:
        min_gap = _prompt_float("Smallest z gap to treat as a separate top group", 1.0)

    selection = runtime.molecule_stage.identify_top_group(structure, z_cutoff=z_cutoff, min_gap=min_gap)
    print("\nDetected top-side group:")
    print(f"- top atoms: {selection.molecule_atom_count}")
    print(f"- bottom atoms: {selection.substrate_atom_count}")
    print(f"- z cutoff: {selection.z_cutoff:.6f} A")
    if selection.gap_size == selection.gap_size:
        print(f"- largest internal z gap: {selection.gap_size:.6f} A")
    print(
        "- center of mass (cartesian): ({0:.6f}, {1:.6f}, {2:.6f}) A".format(
            selection.center_of_mass_cartesian[0],
            selection.center_of_mass_cartesian[1],
            selection.center_of_mass_cartesian[2],
        )
    )
    print(
        "- center of mass (direct): ({0:.6f}, {1:.6f}, {2:.6f})".format(
            selection.center_of_mass_direct[0],
            selection.center_of_mass_direct[1],
            selection.center_of_mass_direct[2],
        )
    )

    output_path = _prompt_text("Optional output POSCAR path", "", allow_empty=True) or None
    if mode == "molecule":
        if coordinate_mode == "direct":
            default_target = "{0:.6f},{1:.6f}".format(selection.center_of_mass_direct[0], selection.center_of_mass_direct[1])
            target_value = _parse_coordinate_vector(_prompt_text("Target center of mass in direct coordinates (u,v[,w])", default_target))
            args = argparse.Namespace(
                poscar=poscar_path,
                output=output_path,
                target_cart=None,
                target_direct=target_value,
                rotate=_prompt_float("Rotation about z in degrees", 0.0),
                z_cutoff=selection.z_cutoff,
                min_gap=min_gap,
                reframe=_prompt_choice(
                    "Reframe the visible periodic image after moving the molecule? none / x / y / xy / xyz",
                    "xy",
                    ["none", "x", "y", "xy", "xyz"],
                ),
            )
        else:
            default_target = "{0:.6f},{1:.6f},{2:.6f}".format(
                selection.center_of_mass_cartesian[0],
                selection.center_of_mass_cartesian[1],
                selection.center_of_mass_cartesian[2],
            )
            target_value = _parse_coordinate_vector(_prompt_text("Target center of mass in cartesian coordinates (x,y[,z])", default_target))
            args = argparse.Namespace(
                poscar=poscar_path,
                output=output_path,
                target_cart=target_value,
                target_direct=None,
                rotate=_prompt_float("Rotation about z in degrees", 0.0),
                z_cutoff=selection.z_cutoff,
                min_gap=min_gap,
                reframe=_prompt_choice(
                    "Reframe the visible periodic image after moving the molecule? none / x / y / xy / xyz",
                    "xy",
                    ["none", "x", "y", "xy", "xyz"],
                ),
            )
        _run_molecule(args)
        return

    if coordinate_mode == "direct":
        shift_value = _parse_coordinate_vector(
            _prompt_text("Upper-layer shift in direct coordinates (du,dv[,dw])", "0.0,0.0")
        )
        args = argparse.Namespace(
            poscar=poscar_path,
            output=output_path,
            shift_cart=None,
            shift_direct=shift_value,
            z_cutoff=selection.z_cutoff,
            min_gap=min_gap,
        )
    else:
        shift_value = _parse_coordinate_vector(
            _prompt_text("Upper-layer shift in cartesian coordinates (dx,dy[,dz])", "0.0,0.0")
        )
        args = argparse.Namespace(
            poscar=poscar_path,
            output=output_path,
            shift_cart=shift_value,
            shift_direct=None,
            z_cutoff=selection.z_cutoff,
            min_gap=min_gap,
        )
    _run_layer(args)


def _interactive_find_then_maybe_make() -> None:
    runtime = _runtime_modules()
    print(f"\n{APP_NAME} interactive workflow")
    print("Let's start by searching for commensurate candidates.")
    poscar_a = _prompt_text("Path to first POSCAR", _default_input_poscar("mos2.vasp"))
    poscar_b = _prompt_text("Path to second POSCAR", _default_input_poscar("mos2.vasp"))
    poscar_a = _resolve_existing_path(poscar_a, [INPUT_DIR, Path(".")])
    poscar_b = _resolve_existing_path(poscar_b, [INPUT_DIR, Path(".")])
    bottom_path, top_path = _resolve_bottom_choice(poscar_a, poscar_b, None)

    top_structure = runtime.io_mod.read_poscar(top_path)
    bottom_structure = runtime.io_mod.read_poscar(bottom_path)
    symmetry_top, symmetry_bottom, symmetry_lcm = runtime.lattice_backend.combined_symmetry_limit(
        top_structure.lattice,
        bottom_structure.lattice,
    )

    print(
        f"Detected symmetry angles: top={symmetry_top} deg, bottom={symmetry_bottom} deg, "
        f"LCM search limit={symmetry_lcm} deg"
    )

    nindex = _prompt_int("Max integer span nindex", 12)
    suggested = _suggest_find_defaults(top_structure.lattice, bottom_structure.lattice)
    print("\nRecommended finder settings:")
    print("- angles are in degrees")
    print("- absolute length mismatch is in angstrom")
    print("- strain-style values are fractions, so 0.0100 means 1.00%")
    print(f"- profile: {suggested['profile']}")
    print(f"- search window: {float(suggested['min_angle']):.1f} -> {float(suggested['max_angle']):.1f} degrees")
    print(f"- fast angle shortlist length mismatch: {float(suggested['angle_length_tolerance']):.2e} angstrom")
    print(f"- fast angle shortlist relative mismatch: {_format_fraction(float(suggested['angle_strain_tolerance']))}")
    print(f"- vector-pair matching tolerance: {_format_fraction(float(suggested['vector_tolerance']))}")
    print(f"- vector-pair relative length mismatch: {_format_fraction(float(suggested['vector_strain_tolerance']))}")
    print(f"- candidate build tolerance: {_format_fraction(float(suggested['candidate_tolerance']))}")
    print(f"- final strain cutoff ({suggested['strain_layer']}): {_format_fraction(float(suggested['strain_tolerance']))}")
    print(f"- max atoms: {int(suggested['max_atoms'])}")
    print(f"- output folder: {suggested['output_root']}")
    use_defaults = _prompt_yes_no("Use these recommended finder settings?", True)
    if use_defaults:
        min_angle = float(suggested["min_angle"])
        max_angle = float(suggested["max_angle"])
        explicit_angles_raw = None
        angle_length_tolerance = float(suggested["angle_length_tolerance"])
        angle_strain_tolerance = float(suggested["angle_strain_tolerance"])
        angle_merge_tolerance = float(suggested["angle_merge_tolerance"])
        vector_tolerance = float(suggested["vector_tolerance"])
        vector_strain_tolerance = float(suggested["vector_strain_tolerance"])
        candidate_tolerance = float(suggested["candidate_tolerance"])
        strain_tolerance = float(suggested["strain_tolerance"])
        strain_layer = str(suggested["strain_layer"])
        max_atoms = int(suggested["max_atoms"])
        top_rows = int(suggested["top_rows"])
        output_root = str(suggested["output_root"])
        matrix_values_raw = None
        matrix_layer = "either"
        matrix_match_mode = "absolute"
    else:
        min_angle = _prompt_float("Minimum angle in degrees", 0.0)
        max_angle = _prompt_float("Maximum angle in degrees", float(symmetry_lcm))
        explicit_angles_raw = _prompt_text(
            "Explicit comma-separated angles in degrees (leave blank to let CELLSTINE shortlist them first)",
            "",
            allow_empty=True,
        )
        angle_length_tolerance = _prompt_float(
            "Fast angle shortlist absolute length mismatch in angstrom",
            float(suggested["angle_length_tolerance"]),
        )
        angle_strain_tolerance = _prompt_float(
            "Fast angle shortlist relative length mismatch as a fraction (0.01 = 1%)",
            float(suggested["angle_strain_tolerance"]),
        )
        angle_merge_tolerance = _prompt_float(
            "Merge nearby shortlist angles within this many degrees",
            float(suggested["angle_merge_tolerance"]),
        )
        vector_tolerance = _prompt_float(
            "Vector-pair matching tolerance as a fraction (0.01 = 1%)",
            float(suggested["vector_tolerance"]),
        )
        vector_strain_tolerance = _prompt_float(
            "Vector-pair relative length mismatch as a fraction (0.01 = 1%)",
            float(suggested["vector_strain_tolerance"]),
        )
        candidate_tolerance = _prompt_float(
            "Candidate build tolerance as a fraction (0.01 = 1%)",
            float(suggested["candidate_tolerance"]),
        )
        strain_tolerance = _prompt_optional_float(
            "Final strain cutoff as a fraction (0.01 = 1%, leave blank to skip)",
            float(suggested["strain_tolerance"]),
        )
        strain_layer = _prompt_text(
            "Which strain column should that final cutoff use? avg / 1 / 2",
            str(suggested["strain_layer"]),
        )
        max_atoms = _prompt_int("Maximum total atoms", int(suggested["max_atoms"]))
        top_rows = _prompt_int("How many candidate rows should be printed? (-1 for all, 0 for none)", int(suggested["top_rows"]))
        output_root = _prompt_text("Folder for saved finder results", str(suggested["output_root"]))
        matrix_values_raw = _prompt_text(
            "Optional matrix values to require (four integers, any order; leave blank to skip)",
            "",
            allow_empty=True,
        ) or None
        matrix_layer = "either"
        matrix_match_mode = "absolute"
        if matrix_values_raw is not None:
            _parse_matrix_values(matrix_values_raw)
            matrix_layer = _prompt_choice(
                "Apply that matrix-value filter to layer 1, layer 2, either, or both?",
                "either",
                ["either", "1", "2", "both"],
            )
            matrix_match_mode = _prompt_choice(
                "Compare those matrix values by absolute value or exact signed value?",
                "absolute",
                ["absolute", "exact"],
            )

    args = argparse.Namespace(
        poscar_a=poscar_a,
        poscar_b=poscar_b,
        bottom="a" if bottom_path == poscar_a else "b",
        nindex=nindex,
        min_angle=min_angle,
        max_angle=max_angle,
        angles=explicit_angles_raw or None,
        angle_step=0.1,
        angle_length_tolerance=angle_length_tolerance,
        angle_strain_tolerance=angle_strain_tolerance,
        angle_merge_tolerance=angle_merge_tolerance,
        vector_tolerance=vector_tolerance,
        vector_strain_tolerance=vector_strain_tolerance,
        candidate_tolerance=candidate_tolerance,
        strain_tolerance=strain_tolerance,
        strain_layer=strain_layer,
        min_atoms=None,
        max_atoms=max_atoms,
        no_dedupe=False,
        unique_strain_tolerance=1e-4,
        unique_ratio_tolerance=1e-5,
        matrix_values=matrix_values_raw,
        matrix_layer=matrix_layer,
        matrix_match_mode=matrix_match_mode,
        output_root=output_root,
        top=top_rows,
    )
    find_run = _run_find(args)

    if not find_run.candidates:
        return
    if _prompt_choice("Next step: generate or close", "generate", ["generate", "close"]) == "close":
        return
    _interactive_make_from_results(str(find_run.dat_path))


def _interactive_make() -> None:
    latest_results = _latest_results_file()
    results_path = _prompt_text("Path to results .dat file", latest_results or str(RUNS_DIR / "latest.dat"))
    _interactive_make_from_results(results_path)


def interactive_main() -> None:
    print(f"=== {APP_NAME} | {APP_EXPANSION} | Made by Sarwin Chandran ===")
    print("1. Search for commensurate candidates")
    print("2. Generate the commensurate superlattice from a saved search")
    print("3. Move a top molecule or shift the upper layer in a stacked POSCAR")
    print("Advanced CLI workflows such as findn, maken, surface, and visualize are available through --help.")

    choice = _prompt_text("Choose workflow", "1")
    if choice == "1":
        _interactive_find_then_maybe_make()
        return
    if choice == "2":
        _interactive_make()
        return
    if choice == "3":
        _interactive_move_top_group()
        return
    raise ValueError("Please choose workflow 1, 2, or 3.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            f"{APP_NAME}: search commensurate matches, generate the commensurate superlattice, "
            "and adjust stacked POSCAR structures."
        ),
        epilog=(
            "Examples:\n"
            "  python moire_cli.py\n"
            "  python moire_cli.py find input/mos2.vasp input/mos2.vasp --nindex 12\n"
            "  python moire_cli.py findn input/substrate.vasp input/layer_b.vasp input/layer_c.vasp --nindex 12\n"
            "  python moire_cli.py make runs/<run_id>.dat --index 1\n"
            "  python moire_cli.py maken runs/<run_id>.json --index 1 --interlayer 3.35 --interlayer 3.35\n"
            "  python moire_cli.py surface input/bulk.vasp --miller 1,1,0 --layers 6 --vacuum 15\n"
            "  python moire_cli.py visualize runs/<run_id>.dat --index 1,2,3\n"
            "  python moire_cli.py molecule output/stacked.vasp --target-direct 0.5,0.5\n"
        ),
        formatter_class=_HelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    find_parser = subparsers.add_parser(
        "find",
        help="advanced mode: search for commensurate superlattice candidates between two POSCAR files",
        description="Search two POSCAR files for commensurate superlattice candidates.",
        epilog=(
            "Units:\n"
            "  angles are in degrees\n"
            "  angle-length-tolerance is in angstrom\n"
            "  strain and mismatch tolerances are fractions (0.01 = 1 percent)\n\n"
            "Examples:\n"
            "  python moire_cli.py find input/mos2.vasp input/mos2.vasp --nindex 12\n"
            "  python moire_cli.py find input/a.vasp input/b.vasp --angles 13.2,21.8\n"
        ),
        formatter_class=_HelpFormatter,
    )
    find_parser.add_argument("poscar_a", help="path to the first POSCAR input, typically from input/")
    find_parser.add_argument("poscar_b", help="path to the second POSCAR input, typically from input/")
    find_parser.add_argument("--bottom", type=str, default=None, help="which input should sit on the bottom side in the final stack: a or b")
    find_parser.add_argument("--nindex", type=int, default=12, help="search integer spans from -nindex to +nindex")
    find_parser.add_argument("--min-angle", type=float, default=None, help="minimum twist angle in degrees")
    find_parser.add_argument("--max-angle", type=float, default=None, help="maximum twist angle in degrees; defaults to the symmetry limit")
    find_parser.add_argument("--angles", type=str, default=None, help="comma-separated explicit twist angles in degrees")
    find_parser.add_argument("--angle-step", type=float, default=0.1, help="fallback scan step in degrees when no shortlist is found")
    find_parser.add_argument("--angle-length-tolerance", type=float, default=None, help="absolute length tolerance for the angle shortlist, in angstrom")
    find_parser.add_argument("--angle-strain-tolerance", type=float, default=None, help="relative length mismatch allowed in the angle shortlist, as a fraction (0.01 = 1 percent)")
    find_parser.add_argument("--angle-merge-tolerance", type=float, default=None, help="merge nearby angles within this tolerance, in degrees")
    find_parser.add_argument("--vector-tolerance", type=float, default=None, help="relative geometric tolerance used when pairing vectors, as a fraction (0.01 = 1 percent)")
    find_parser.add_argument("--vector-strain-tolerance", type=float, default=None, help="relative vector length mismatch allowed during pairing, as a fraction (0.01 = 1 percent)")
    find_parser.add_argument("--candidate-tolerance", type=float, default=None, help="relative tolerance used when assembling candidate supercells, as a fraction (0.01 = 1 percent)")
    find_parser.add_argument("--strain-tolerance", type=float, default=None, help="final strain cutoff, as a fraction (0.01 = 1 percent), applied to avg, 1, or 2")
    find_parser.add_argument("--strain-layer", choices=["avg", "1", "2"], default="avg", help="which final strain column the --strain-tolerance filter should use")
    find_parser.add_argument("--min-atoms", type=int, default=None, help="minimum total atom count to keep")
    find_parser.add_argument("--max-atoms", type=int, default=None, help="maximum total atom count to keep")
    find_parser.add_argument("--no-dedupe", action="store_true", help="keep near-duplicate candidates instead of collapsing them")
    find_parser.add_argument("--unique-strain-tolerance", type=float, default=1e-4, help="deduplication tolerance on strain values, as a fraction")
    find_parser.add_argument("--unique-ratio-tolerance", type=float, default=1e-5, help="deduplication tolerance on ratio comparisons, as a fraction")
    find_parser.add_argument("--matrix-values", type=_parse_matrix_values, default=None, help="optional four matrix values to require for a 2x2 supercell matrix, in any order, for example 1,2,3,4")
    find_parser.add_argument("--matrix-layer", choices=["1", "2", "either", "both"], default="either", help="which layer matrix the --matrix-values filter should match")
    find_parser.add_argument("--matrix-match-mode", choices=["absolute", "exact"], default="absolute", help="compare matrix values by absolute value or by exact signed value")
    find_parser.add_argument("--top-c-repeat", type=int, default=1, help="repeat the top input structure along c this many times before counting atoms and building the supercell")
    find_parser.add_argument("--bottom-c-repeat", type=int, default=1, help="repeat the bottom input structure along c this many times before counting atoms and building the supercell")
    find_parser.add_argument("--workers", type=int, default=1, help="number of worker processes for the angle search; use 1 for the original single-threaded workflow")
    find_parser.add_argument("--output-root", type=str, default="runs", help="folder where finder results should be saved; defaults to runs/")
    find_parser.add_argument("--top", type=int, default=10, help="number of candidate rows to print; use -1 for all and 0 for none")
    find_parser.set_defaults(func=_run_find)

    findn_parser = subparsers.add_parser(
        "findn",
        help="advanced mode: search for N-layer commensuration using one bottom reference layer and multiple rotated upper layers",
        description="Search one bottom POSCAR plus one or more upper POSCAR files for shared reference-layer commensurate superlattice candidates.",
        epilog=(
            "Units:\n"
            "  all angles are in degrees\n"
            "  angle-length-tolerance is in angstrom\n"
            "  strain and mismatch tolerances are fractions (0.01 = 1 percent)\n\n"
            "Examples:\n"
            "  python moire_cli.py findn input/substrate.vasp input/layer_b.vasp input/layer_c.vasp --nindex 12\n"
            "  python moire_cli.py findn input/substrate.vasp input/layer_b.vasp input/layer_c.vasp input/layer_d.vasp --angles 13.2 --angles 21.8 --angles 5.5\n"
        ),
        formatter_class=_HelpFormatter,
    )
    findn_parser.add_argument("bottom_poscar", help="path to the bottom reference POSCAR input, typically from input/")
    findn_parser.add_argument("upper_poscars", nargs="+", help="one or more upper POSCAR inputs in stack order, typically from input/")
    findn_parser.add_argument("--nindex", type=int, default=12, help="search integer spans from -nindex to +nindex")
    findn_parser.add_argument("--min-angle", dest="min_angles", action="append", type=float, default=None, help="minimum twist angle in degrees; give once to apply to all upper layers, or repeat once per upper layer")
    findn_parser.add_argument("--max-angle", dest="max_angles", action="append", type=float, default=None, help="maximum twist angle in degrees; give once to apply to all upper layers, or repeat once per upper layer")
    findn_parser.add_argument("--angles", action="append", type=str, default=None, help="explicit comma-separated twist angles in degrees; give once to apply to one upper layer, or repeat once per upper layer")
    findn_parser.add_argument("--angle-step", type=float, default=0.1, help="fallback scan step in degrees when no shortlist is found")
    findn_parser.add_argument("--angle-length-tolerance", type=float, default=None, help="absolute length tolerance for the angle shortlist, in angstrom")
    findn_parser.add_argument("--angle-strain-tolerance", type=float, default=None, help="relative length mismatch allowed in the angle shortlist, as a fraction (0.01 = 1 percent)")
    findn_parser.add_argument("--angle-merge-tolerance", type=float, default=None, help="merge nearby angles within this tolerance, in degrees")
    findn_parser.add_argument("--vector-tolerance", type=float, default=None, help="relative geometric tolerance used when pairing vectors, as a fraction (0.01 = 1 percent)")
    findn_parser.add_argument("--vector-strain-tolerance", type=float, default=None, help="relative vector length mismatch allowed during pairing, as a fraction (0.01 = 1 percent)")
    findn_parser.add_argument("--candidate-tolerance", type=float, default=None, help="relative tolerance used when assembling candidate supercells, as a fraction (0.01 = 1 percent)")
    findn_parser.add_argument("--pair-strain-tolerance", type=float, default=None, help="final pairwise strain cutoff applied independently to each upper-layer versus bottom-layer search")
    findn_parser.add_argument("--max-atoms", type=int, default=None, help="maximum total atom count to keep after joining the pairwise searches")
    findn_parser.add_argument("--no-dedupe", action="store_true", help="keep near-duplicate pairwise candidates instead of collapsing them before joining")
    findn_parser.add_argument("--unique-strain-tolerance", type=float, default=1e-4, help="deduplication tolerance on pairwise strain values, as a fraction")
    findn_parser.add_argument("--unique-ratio-tolerance", type=float, default=1e-5, help="deduplication tolerance on pairwise ratio comparisons, as a fraction")
    findn_parser.add_argument("--bottom-c-repeat", type=int, default=1, help="repeat the bottom structure along c this many times before counting atoms and generation")
    findn_parser.add_argument("--upper-c-repeat", action="append", type=int, default=None, help="repeat count along c for upper layers; give once to apply to all, or repeat once per upper layer")
    findn_parser.add_argument("--workers", type=int, default=1, help="number of worker processes used inside each pairwise angle search; use 1 for the original single-threaded workflow")
    findn_parser.add_argument("--output-root", type=str, default="runs", help="folder where N-layer results should be saved; defaults to runs/")
    findn_parser.add_argument("--top", type=int, default=10, help="number of candidate rows to print; use -1 for all and 0 for none")
    findn_parser.set_defaults(func=_run_findn)


    make_parser = subparsers.add_parser(
        "make",
        help="advanced mode: generate the commensurate superlattice from saved results",
        description="Generate the commensurate superlattice from a saved CELLSTINE results file.",
        epilog=(
            "Units:\n"
            "  interlayer distance and zfix are in angstrom\n\n"
            "Examples:\n"
            "  python moire_cli.py make runs/<run_id>.dat --index 1\n"
            "  python moire_cli.py make runs/<run_id>.dat --index 1,2,5-7 --output-dir output\n"
        ),
        formatter_class=_HelpFormatter,
    )
    make_parser.add_argument("results", help="path to the finder results .dat file, typically from runs/")
    make_parser.add_argument("--index", type=str, default=None, help="candidate index spec, required in command mode, e.g. 1 or 1,2,5-7")
    make_parser.add_argument("--interlayer", type=float, default=None, help="desired gap between the substrate top and the adsorbate bottom in angstrom")
    make_parser.add_argument("--output", type=str, default=None, help="explicit output POSCAR filename for a single index")
    make_parser.add_argument("--output-dir", type=str, default=None, help="directory for auto-named generated POSCARs; defaults to output/")
    make_parser.add_argument("--generator-tolerance", type=int, default=1, help="integer padding used while collecting periodic images")
    make_parser.add_argument("--generator-tolerance-float", type=float, default=1e-4, help="floating-point tolerance used while collecting periodic images")
    make_parser.add_argument("--zfix", type=float, default=None, help="optional z cutoff in angstrom for selective dynamics flags")
    make_parser.add_argument("--top-c-repeat", type=int, default=None, help="override the top c-repeat used during generation; defaults to the value stored in the results file or 1")
    make_parser.add_argument("--bottom-c-repeat", type=int, default=None, help="override the bottom c-repeat used during generation; defaults to the value stored in the results file or 1")
    make_parser.add_argument("--workers", type=int, default=1, help="number of worker processes for batch generation; use 1 for the original single-threaded workflow")
    make_parser.set_defaults(func=_run_make)

    maken_parser = subparsers.add_parser(
        "maken",
        help="advanced mode: generate an N-layer commensurate superlattice from saved findn results",
        description="Generate the N-layer commensurate superlattice from a saved CELLSTINE findn results file.",
        epilog=(
            "Units:\n"
            "  interlayer distances and zfix are in angstrom\n\n"
            "Examples:\n"
            "  python moire_cli.py maken runs/<run_id>.json --index 1 --interlayer 3.35 --interlayer 3.35\n"
            "  python moire_cli.py maken runs/<run_id>.json --index 1 --interlayer 3.0 --interlayer 3.2 --interlayer 3.4\n"
        ),
        formatter_class=_HelpFormatter,
    )
    maken_parser.add_argument("results", help="path to the N-layer results .json file, typically from runs/")
    maken_parser.add_argument("--index", type=str, default=None, help="candidate index spec, required in command mode, e.g. 1 or 1,2,5-7")
    maken_parser.add_argument("--interlayer", dest="interlayers", action="append", type=float, default=None, help="gap between consecutive layers in angstrom; give once to apply to all upper layers, or repeat once per upper layer")
    maken_parser.add_argument("--output-dir", type=str, default=None, help="directory for auto-named generated POSCARs; defaults to output/")
    maken_parser.add_argument("--bottom-c-repeat", type=int, default=None, help="override the bottom c-repeat used during generation; defaults to the value stored in the results file or 1")
    maken_parser.add_argument("--upper-c-repeat", action="append", type=int, default=None, help="override upper c-repeats; give once to apply to all upper layers, or repeat once per upper layer")
    maken_parser.add_argument("--zfix", type=float, default=None, help="optional z cutoff in angstrom for selective dynamics flags")
    maken_parser.set_defaults(func=_run_maken)

    molecule_parser = subparsers.add_parser(
        "molecule",
        help="move or rotate a top-side molecule/adsorbate inside a stacked POSCAR",
        description="Move a top-side molecule by its center of mass, optionally rotate it, and reframe the visible periodic image.",
        epilog=(
            "Units:\n"
            "  Cartesian distances are in angstrom\n"
            "  Direct coordinates are fractional lattice coordinates\n\n"
            "Examples:\n"
            "  python moire_cli.py molecule output/stacked.vasp --target-direct 0.5,0.5\n"
            "  python moire_cli.py molecule output/stacked.vasp --target-cart 12.0,8.0,10.5 --rotate 30\n"
        ),
        formatter_class=_HelpFormatter,
    )
    molecule_parser.add_argument("poscar", help="stacked POSCAR containing the substrate and top-side molecule, usually from output/ or runs/")
    molecule_parser.add_argument("--output", type=str, default=None, help="optional output POSCAR path; defaults to output/")
    molecule_parser.add_argument(
        "--target-cart",
        type=_parse_coordinate_vector,
        default=None,
        help="target center-of-mass point in Cartesian coordinates, e.g. 5.0,7.0 or 5.0,7.0,12.5",
    )
    molecule_parser.add_argument(
        "--target-direct",
        type=_parse_coordinate_vector,
        default=None,
        help="target center-of-mass point in Direct coordinates, e.g. 0.5,0.5 or 0.5,0.5,0.35",
    )
    molecule_parser.add_argument(
        "--rotate",
        type=float,
        default=0.0,
        help="rotation angle in degrees about the moved molecule COM around an axis parallel to z",
    )
    molecule_parser.add_argument("--z-cutoff", type=float, default=None, help="explicit z cutoff in angstrom for molecule selection")
    molecule_parser.add_argument("--min-gap", type=float, default=1.0, help="minimum internal z gap in angstrom for automatic molecule detection")
    molecule_parser.add_argument(
        "--reframe",
        type=str,
        default="none",
        help="optional visible-cell reframing axes: none, x, y, z, xy, xyz",
    )
    molecule_parser.set_defaults(func=_run_molecule)

    layer_parser = subparsers.add_parser(
        "layer",
        help="shift the upper layer in a commensurate bilayer or stacked POSCAR",
        description="Shift the upper layer in a bilayer or stacked POSCAR without moving the lower structure.",
        epilog=(
            "Units:\n"
            "  Cartesian distances are in angstrom\n"
            "  Direct coordinates are fractional lattice coordinates\n\n"
            "Examples:\n"
            "  python moire_cli.py layer output/stacked.vasp --shift-direct 0.333,0.667\n"
            "  python moire_cli.py layer output/stacked.vasp --shift-cart 1.2,0.5\n"
        ),
        formatter_class=_HelpFormatter,
    )
    layer_parser.add_argument("poscar", help="stacked POSCAR containing the lower and upper layers, usually from output/ or runs/")
    layer_parser.add_argument("--output", type=str, default=None, help="optional output POSCAR path; defaults to output/")
    layer_parser.add_argument(
        "--shift-cart",
        type=_parse_coordinate_vector,
        default=None,
        help="shift vector in Cartesian coordinates, e.g. 1.2,0.5 or 1.2,0.5,0.0",
    )
    layer_parser.add_argument(
        "--shift-direct",
        type=_parse_coordinate_vector,
        default=None,
        help="shift vector in Direct coordinates, e.g. 0.33,0.67 or 0.33,0.67,0.0",
    )
    layer_parser.add_argument("--z-cutoff", type=float, default=None, help="explicit z cutoff in angstrom for upper-layer selection")
    layer_parser.add_argument("--min-gap", type=float, default=1.0, help="minimum internal z gap in angstrom for automatic upper-layer detection")
    layer_parser.set_defaults(func=_run_layer)

    surface_parser = subparsers.add_parser(
        "surface",
        help="experimental mode: build an orthogonal surface slab from a bulk POSCAR and Miller plane",
        description="Build a slab surface from an orthogonal bulk POSCAR using a Miller plane such as 1,0,0 or 1,1,0.",
        epilog=(
            "Current limitation:\n"
            "  this experimental builder supports orthogonal bulk cells only (cubic, tetragonal, orthorhombic)\n\n"
            "Examples:\n"
            "  python moire_cli.py surface input/bulk.vasp --miller 1,0,0 --layers 6 --vacuum 15\n"
            "  python moire_cli.py surface input/bulk.vasp --miller 1,1,0 --layers 4 --repeat-a 2 --repeat-b 2\n"
        ),
        formatter_class=_HelpFormatter,
    )
    surface_parser.add_argument("bulk_poscar", help="path to the bulk POSCAR input, typically from input/")
    surface_parser.add_argument("--miller", type=_parse_miller, required=True, help="Miller plane as h,k,l, for example 1,1,0")
    surface_parser.add_argument("--layers", type=int, default=4, help="number of repeats along the surface normal to keep in the slab")
    surface_parser.add_argument("--vacuum", type=float, default=15.0, help="vacuum thickness in angstrom added along the slab normal")
    surface_parser.add_argument("--repeat-a", type=int, default=1, help="repeat the oriented slab this many times along the first in-plane axis")
    surface_parser.add_argument("--repeat-b", type=int, default=1, help="repeat the oriented slab this many times along the second in-plane axis")
    surface_parser.add_argument("--output", type=str, default=None, help="optional output POSCAR path; defaults to output/")
    surface_parser.set_defaults(func=_run_surface)

    visualize_parser = subparsers.add_parser(
        "visualize",
        help="build an interactive Plotly HTML viewer for bilayer or N-layer commensurate results",
        description="Create an interactive HTML viewer that snaps through commensurate twist-angle frames and draws the commensurate unit cell.",
        epilog=(
            "Examples:\n"
            "  python moire_cli.py visualize runs/<run_id>.dat --index 1,2,3\n"
            "  python moire_cli.py visualize runs/<run_id>.json --index 1,2\n"
        ),
        formatter_class=_HelpFormatter,
    )
    visualize_parser.add_argument("results", help="path to a bilayer .dat results file or an N-layer .json results file")
    visualize_parser.add_argument("--index", type=str, default=None, help="optional candidate index spec, e.g. 1 or 1,2,5-7; defaults to all rows in the results file")
    visualize_parser.add_argument("--interlayer", type=float, default=3.35, help="bilayer gap in angstrom used when visualizing a bilayer results file")
    visualize_parser.add_argument("--interlayer-bottom-middle", type=float, default=3.35, help="bottom-to-first-upper gap in angstrom used when visualizing an N-layer results file")
    visualize_parser.add_argument("--interlayer-middle-top", type=float, default=3.35, help="second gap used when visualizing a 3-layer results file")
    visualize_parser.add_argument("--bottom-c-repeat", type=int, default=None, help="override the bottom c-repeat used during visualization")
    visualize_parser.add_argument("--middle-c-repeat", type=int, default=None, help="override the middle c-repeat used during visualization")
    visualize_parser.add_argument("--top-c-repeat", type=int, default=None, help="override the top c-repeat used during visualization")
    visualize_parser.add_argument("--output", type=str, default=None, help="optional output HTML path; defaults to output/")
    visualize_parser.set_defaults(func=_run_visualize)

    return parser


def main() -> None:
    if len(sys.argv) == 1:
        interactive_main()
        return

    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "command", None) is None:
        interactive_main()
        return

    if args.command == "make" and args.interlayer is None:
        args.interlayer = 3.35

    args.func(args)


if __name__ == "__main__":
    main()
