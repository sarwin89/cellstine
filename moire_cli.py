#!/usr/bin/env python3
"""Interactive user-facing CLI for moire find/make stages.

Made by Sarwin Chandran.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from moire import find as find_stage
from moire import finder as finder_backend
from moire import generator as generator_backend
from moire import io as io_mod
from moire import lattice as lattice_backend
from moire import make as make_stage


def _suggest_find_defaults(top_lattice, bottom_lattice) -> dict[str, float | int | str]:
    top_a, top_b, _ = lattice_backend.in_plane_lengths_and_angle(top_lattice)
    bottom_a, bottom_b, _ = lattice_backend.in_plane_lengths_and_angle(bottom_lattice)
    top_scale = 0.5 * (top_a + top_b)
    bottom_scale = 0.5 * (bottom_a + bottom_b)
    primitive_mismatch = abs(top_scale - bottom_scale) / max(0.5 * (top_scale + bottom_scale), 1e-12)
    _, _, symmetry_lcm = lattice_backend.combined_symmetry_limit(top_lattice, bottom_lattice)

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
        print("Please enter a value.")


def _prompt_int(prompt_text: str, default_value: int) -> int:
    while True:
        user_text = _prompt_text(prompt_text, str(default_value))
        try:
            return int(user_text)
        except ValueError:
            print("Please enter a whole number.")


def _prompt_float(prompt_text: str, default_value: float) -> float:
    while True:
        user_text = _prompt_text(prompt_text, str(default_value))
        try:
            return float(user_text)
        except ValueError:
            print("Please enter a number.")


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
        print("Please answer y or n.")


def _parse_angles(raw: str | None) -> List[float] | None:
    if not raw:
        return None
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


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
    run_root = Path("runs")
    if not run_root.exists():
        return None
    candidates = sorted(run_root.glob("*/find_results.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if candidates:
        return str(candidates[0])
    dat_candidates = sorted(run_root.glob("*/find_results.dat"), key=lambda path: path.stat().st_mtime, reverse=True)
    if dat_candidates:
        return str(dat_candidates[0])
    return None


def _print_find_summary(run: find_stage.FindRun, bottom_path: str, top_path: str, top_count: int) -> None:
    print("\n=== Moire Find Stage (Made by Sarwin Chandran) ===")
    print(f"Bottom layer        : {bottom_path}")
    print(f"Top layer           : {top_path}")
    print(f"Symmetry(top,bottom): ({run.symmetry_top}, {run.symmetry_bottom})")
    print(f"LCM angle limit     : {run.symmetry_lcm:.0f} deg")
    print(f"Search window       : {run.search_min_angle:.4f} -> {run.search_max_angle:.4f} deg")
    print(f"Angles checked      : {len(run.angle_values)}")
    print(f"Candidates found    : {len(run.candidates)}")
    print(f"Saved JSON          : {run.json_path}")
    print(f"Saved Markdown      : {run.markdown_path}")
    print(f"Saved DAT           : {run.dat_path}")

    if run.candidates and top_count > 0:
        print("\nTop candidates:")
        print(finder_backend.format_results_table(run.candidates, limit=top_count))


def _print_make_summary(run: make_stage.MakeRun) -> None:
    print("\n=== Moire Make Stage (Made by Sarwin Chandran) ===")
    print(f"Selected index : {run.selected_index}")
    print(f"Angle (deg)    : {run.angle_deg:.4f}")
    print(f"Total atoms    : {run.total_atoms}")
    print(f"Output POSCAR  : {run.output_path}")


def _run_find(args: argparse.Namespace) -> find_stage.FindRun:
    bottom_path, top_path = _resolve_bottom_choice(args.poscar_a, args.poscar_b, args.bottom)
    top_structure = io_mod.read_poscar(top_path)
    bottom_structure = io_mod.read_poscar(bottom_path)
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

    run = find_stage.run_find(
        top_poscar=top_path,
        bottom_poscar=bottom_path,
        top_lattice=top_structure.lattice,
        bottom_lattice=bottom_structure.lattice,
        top_atoms=top_structure.natoms,
        bottom_atoms=bottom_structure.natoms,
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
        output_root=args.output_root,
    )

    _print_find_summary(run, bottom_path, top_path, args.top)
    return run


def _run_make(args: argparse.Namespace) -> make_stage.MakeRun:
    _, _, records, _ = generator_backend.parse_results(args.results)
    if not records:
        raise ValueError("results file has no candidate records")

    index = 1 if args.index is None else args.index
    interlayer = 3.35 if args.interlayer is None else args.interlayer

    run = make_stage.generate_from_results(
        args.results,
        index=index,
        interlayer_distance=interlayer,
        output_path=args.output,
        tolerance=args.generator_tolerance,
        tolerance_float=args.generator_tolerance_float,
        zfix=args.zfix,
    )
    _print_make_summary(run)
    return run


def _interactive_find_then_maybe_make(*, offer_make: bool) -> None:
    print("\nMoire CLI interactive workflow")
    poscar_a = _prompt_text("Path to first POSCAR", "mos2.vasp")
    poscar_b = _prompt_text("Path to second POSCAR", "mos2.vasp")
    bottom_path, top_path = _resolve_bottom_choice(poscar_a, poscar_b, None)

    top_structure = io_mod.read_poscar(top_path)
    bottom_structure = io_mod.read_poscar(bottom_path)
    symmetry_top, symmetry_bottom, symmetry_lcm = lattice_backend.combined_symmetry_limit(
        top_structure.lattice,
        bottom_structure.lattice,
    )

    print(
        f"Detected symmetry angles: top={symmetry_top} deg, bottom={symmetry_bottom} deg, "
        f"LCM search limit={symmetry_lcm} deg"
    )

    nindex = _prompt_int("Max integer span nindex", 12)
    suggested = _suggest_find_defaults(top_structure.lattice, bottom_structure.lattice)
    print("\nRecommended defaults:")
    print(f"- profile: {suggested['profile']}")
    print(f"- search window: {float(suggested['min_angle']):.1f} -> {float(suggested['max_angle']):.1f} deg")
    print(f"- angle strain tolerance: {float(suggested['angle_strain_tolerance']):.4f}")
    print(f"- vector tolerance: {float(suggested['vector_tolerance']):.4f}")
    print(f"- vector strain tolerance: {float(suggested['vector_strain_tolerance']):.4f}")
    print(f"- final strain cutoff: {float(suggested['strain_tolerance']):.4f}")
    print(f"- max atoms: {int(suggested['max_atoms'])}")
    print(f"- output folder: {suggested['output_root']}")

    use_defaults = _prompt_yes_no("Use recommended defaults and continue?", True)
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
    else:
        min_angle = _prompt_float("Minimum angle in degrees", 0.0)
        max_angle = _prompt_float("Maximum angle in degrees", float(symmetry_lcm))
        explicit_angles_raw = _prompt_text(
            "Explicit comma-separated angles (leave blank to auto-find exact commensurate angles)",
            "",
            allow_empty=True,
        )
        angle_length_tolerance = _prompt_float("Absolute length tolerance for angle shortlist", float(suggested["angle_length_tolerance"]))
        angle_strain_tolerance = _prompt_float("Relative length-mismatch tolerance for angle shortlist", float(suggested["angle_strain_tolerance"]))
        angle_merge_tolerance = _prompt_float("Merge tolerance for near-identical angles", float(suggested["angle_merge_tolerance"]))
        vector_tolerance = _prompt_float("Vector coincidence tolerance", float(suggested["vector_tolerance"]))
        vector_strain_tolerance = _prompt_float("Relative vector-length mismatch tolerance", float(suggested["vector_strain_tolerance"]))
        candidate_tolerance = _prompt_float("Candidate vector-pair tolerance", vector_tolerance)
        strain_tolerance = _prompt_optional_float("Final strain cutoff", float(suggested["strain_tolerance"]))
        strain_layer = _prompt_text("Which strain column to filter on? avg / 1 / 2", str(suggested["strain_layer"]))
        max_atoms = _prompt_int("Maximum total atoms", int(suggested["max_atoms"]))
        top_rows = _prompt_int("How many top rows should be printed", int(suggested["top_rows"]))
        output_root = _prompt_text("Output folder for find results", str(suggested["output_root"]))

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
        output_root=output_root,
        top=top_rows,
    )
    find_run = _run_find(args)

    if not offer_make or not find_run.candidates:
        return
    if not _prompt_yes_no("Generate a stacked structure from these results now?", False):
        return

    make_args = argparse.Namespace(
        results=str(find_run.json_path),
        index=_prompt_int("Candidate index to generate", 1),
        interlayer=_prompt_float("Interlayer distance in angstrom", 3.35),
        output=_prompt_text("Optional output POSCAR path (leave blank for auto-name)", "", allow_empty=True) or None,
        generator_tolerance=_prompt_int("Integer padding tolerance for generator", 1),
        generator_tolerance_float=_prompt_float("Floating tolerance for generator", 1e-4),
        zfix=_prompt_optional_float("Optional zfix cutoff (blank to skip)", None),
    )
    _run_make(make_args)


def _interactive_make() -> None:
    latest_results = _latest_results_file()
    results_path = _prompt_text("Path to find_results.json or find_results.dat", latest_results)
    _, _, records, _ = generator_backend.parse_results(results_path)
    print(f"Loaded {len(records)} candidate rows from {results_path}")

    args = argparse.Namespace(
        results=results_path,
        index=_prompt_int("Candidate index to generate", 1),
        interlayer=_prompt_float("Interlayer distance in angstrom", 3.35),
        output=_prompt_text("Optional output POSCAR path (leave blank for auto-name)", "", allow_empty=True) or None,
        generator_tolerance=_prompt_int("Integer padding tolerance for generator", 1),
        generator_tolerance_float=_prompt_float("Floating tolerance for generator", 1e-4),
        zfix=_prompt_optional_float("Optional zfix cutoff (blank to skip)", None),
    )
    _run_make(args)


def interactive_main() -> None:
    print("=== Moire CLI | Made by Sarwin Chandran ===")
    print("1. Find commensurate angles and supercells")
    print("2. Make a supercell from saved finder results")
    print("3. Find first, then generate immediately")
    print("Advanced command flags are still available through --help.")

    choice = _prompt_text("Choose workflow", "1")
    if choice == "1":
        _interactive_find_then_maybe_make(offer_make=False)
        return
    if choice == "2":
        _interactive_make()
        return
    if choice == "3":
        _interactive_find_then_maybe_make(offer_make=True)
        return
    raise ValueError("workflow choice must be 1, 2, or 3")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Moire toolkit CLI with interactive workflow plus optional advanced subcommands.",
    )
    subparsers = parser.add_subparsers(dest="command")

    find_parser = subparsers.add_parser("find", help="advanced mode: find commensurate angles and superstructure candidates")
    find_parser.add_argument("poscar_a", help="first POSCAR input")
    find_parser.add_argument("poscar_b", help="second POSCAR input")
    find_parser.add_argument("--bottom", type=str, default=None, help="which POSCAR is bottom layer: a or b")
    find_parser.add_argument("--nindex", type=int, default=12, help="integer span from -nindex to nindex")
    find_parser.add_argument("--min-angle", type=float, default=None, help="minimum angle to consider")
    find_parser.add_argument("--max-angle", type=float, default=None, help="optional max angle cap")
    find_parser.add_argument("--angles", type=str, default=None, help="comma-separated explicit angles")
    find_parser.add_argument("--angle-step", type=float, default=0.1, help="fallback scan step when shortlist is empty")
    find_parser.add_argument("--angle-length-tolerance", type=float, default=None)
    find_parser.add_argument("--angle-strain-tolerance", type=float, default=None)
    find_parser.add_argument("--angle-merge-tolerance", type=float, default=None)
    find_parser.add_argument("--vector-tolerance", type=float, default=None)
    find_parser.add_argument("--vector-strain-tolerance", type=float, default=None)
    find_parser.add_argument("--candidate-tolerance", type=float, default=None)
    find_parser.add_argument("--strain-tolerance", type=float, default=None)
    find_parser.add_argument("--strain-layer", choices=["avg", "1", "2"], default="avg")
    find_parser.add_argument("--min-atoms", type=int, default=None)
    find_parser.add_argument("--max-atoms", type=int, default=None)
    find_parser.add_argument("--no-dedupe", action="store_true")
    find_parser.add_argument("--unique-strain-tolerance", type=float, default=1e-4)
    find_parser.add_argument("--unique-ratio-tolerance", type=float, default=1e-5)
    find_parser.add_argument("--output-root", type=str, default="runs")
    find_parser.add_argument("--top", type=int, default=10, help="top candidates to print")
    find_parser.set_defaults(func=_run_find)

    make_parser = subparsers.add_parser("make", help="advanced mode: build final stacked structure from saved results")
    make_parser.add_argument("results", help="path to find_results.json or find_results.dat")
    make_parser.add_argument("--index", type=int, default=None, help="1-based candidate index")
    make_parser.add_argument("--interlayer", type=float, default=None, help="top-layer z offset in angstrom")
    make_parser.add_argument("--output", type=str, default=None, help="output POSCAR filename")
    make_parser.add_argument("--generator-tolerance", type=int, default=1)
    make_parser.add_argument("--generator-tolerance-float", type=float, default=1e-4)
    make_parser.add_argument("--zfix", type=float, default=None)
    make_parser.set_defaults(func=_run_make)

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

    if args.command == "make":
        if args.index is None:
            args.index = 1
        if args.interlayer is None:
            args.interlayer = 3.35

    args.func(args)


if __name__ == "__main__":
    main()
