"""Guided interactive launcher for the grouped CELLSTINE CLI."""

from __future__ import annotations

from .main import dispatch_namespace
from .parsers import build_parser


def _prompt(prompt: str, default: str | None = None) -> str:
    shown = f" [{default}]" if default not in {None, ''} else ""
    while True:
        answer = input(f"{prompt}{shown}: ").strip()
        if answer:
            return answer
        if default is not None:
            return default
        print("Please enter a value.")


def run_interactive() -> int:
    parser = build_parser()
    print("CELLSTINE interactive mode")
    workflow = _prompt("Choose workflow: moire, adsorbate, interface", "moire").lower()
    if workflow == "moire":
        stage = _prompt("Choose stage: find, findn, make, maken, translate, translaten, visualize", "find").lower()
        if stage == "find":
            argv = [
                "moire", "find",
                _prompt("Top POSCAR path"),
                _prompt("Bottom POSCAR path"),
                "--nindex", _prompt("nindex", "12"),
            ]
        elif stage == "findn":
            bottom = _prompt("Bottom POSCAR path")
            uppers = _prompt("Upper POSCAR paths separated by commas")
            argv = ["moire", "findn", bottom, *[item.strip() for item in uppers.split(",") if item.strip()]]
        elif stage == "make":
            argv = [
                "moire", "make",
                _prompt("Results file (.dat or manifest.json)"),
                "--indexes", _prompt("Indexes", "1"),
                "--interlayer-distance", _prompt("Interlayer distance in angstrom", "3.35"),
            ]
        elif stage == "maken":
            argv = [
                "moire", "maken",
                _prompt("Results file (.json or manifest.json)"),
                "--indexes", _prompt("Indexes", "1"),
                "--interlayers", _prompt("Interlayer distances", "3.35,3.35"),
            ]
        elif stage in {"translate", "translaten"}:
            argv = [
                "moire", stage,
                _prompt("Stacked POSCAR path"),
                "--shift-direct", _prompt("Shift in direct coordinates u,v[,w]", "0.0,0.0"),
            ]
        else:
            argv = ["moire", "visualize", _prompt("Results file")]
    elif workflow == "adsorbate":
        stage = _prompt("Choose stage: place, move, assemble, visualize", "place").lower()
        if stage == "place":
            argv = [
                "adsorbate", "place",
                _prompt("Substrate path"),
                _prompt("Molecule path"),
                "--site-type", _prompt("Site type", "top"),
            ]
        elif stage == "move":
            argv = [
                "adsorbate", "move",
                _prompt("Stacked POSCAR path"),
                "--target-direct", _prompt("Target COM in direct coordinates u,v[,w]", "0.5,0.5"),
            ]
        elif stage == "assemble":
            argv = [
                "adsorbate", "assemble",
                _prompt("Substrate path"),
                "--a-length", _prompt("Target a length in angstrom"),
                "--b-length", _prompt("Target b length in angstrom"),
                "--angle", _prompt("Target angle in degrees", "60"),
            ]
        else:
            argv = ["adsorbate", "visualize", _prompt("Structure POSCAR path")]
    else:
        stage = _prompt("Choose stage: surface, sites, build, match, visualize", "surface").lower()
        if stage == "surface":
            argv = [
                "interface", "surface",
                _prompt("Bulk POSCAR path"),
                "--miller", _prompt("Miller indices", "1,1,1"),
                "--layers", _prompt("Number of layers", "4"),
            ]
        elif stage == "sites":
            argv = ["interface", "sites", _prompt("Slab POSCAR path")]
        elif stage == "build":
            argv = [
                "interface", "build",
                _prompt("Bottom slab or bulk path"),
                _prompt("Top slab or bulk path"),
            ]
        elif stage == "match":
            argv = [
                "interface", "match",
                _prompt("Bottom bulk path"),
                _prompt("Top bulk path"),
            ]
        else:
            argv = ["interface", "visualize", _prompt("Structure POSCAR path")]

    namespace = parser.parse_args(argv)
    return dispatch_namespace(namespace)
