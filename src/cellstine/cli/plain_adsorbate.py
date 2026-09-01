"""Dependency-free parser group for adsorbate commands."""

from __future__ import annotations

import argparse

from .argtypes import (
    HelpFormatter,
    parse_float_vector,
    parse_int_matrix,
    parse_nonnegative_float,
    parse_positive_float,
)


def add_adsorbate_group(groups: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the dependency-free ``adsorbate`` command group."""

    adsorbate = groups.add_parser(
        "adsorbate",
        formatter_class=HelpFormatter,
        help="molecule on substrate workflows",
        description="Molecule-on-substrate workflows. Run `cellstine adsorbate` with no stage to start the guided workflow.",
    )
    adsorbate_sub = adsorbate.add_subparsers(dest="stage")

    ads_place = adsorbate_sub.add_parser("place", formatter_class=HelpFormatter, help="place a molecule on a substrate site")
    ads_place.add_argument("substrate_poscar")
    ads_place.add_argument("molecule_poscar")
    ads_place.add_argument("--substrate-kind", choices=["bulk", "substrate", "patch", "surface", "slab"], default="substrate")
    ads_place.add_argument("--miller", default="1,1,1", help="surface Miller indices for bulk inputs")
    ads_place.add_argument("--layers", type=int, default=4)
    ads_place.add_argument("--vacuum", type=float, default=15.0)
    ads_place.add_argument("--substrate-repeat-a", type=int, default=1, help="repeat a bulk-derived substrate along surface a before placement")
    ads_place.add_argument("--substrate-repeat-b", type=int, default=1, help="repeat a bulk-derived substrate along surface b before placement")
    ads_place.add_argument("--substrate-supercell-matrix", type=parse_int_matrix, default=None, help="2x2 in-plane matrix for a bulk-derived substrate, e.g. 1,1,0,2")
    ads_place.add_argument("--auto-repeat-substrate", action="store_true", help="enlarge the selected substrate if the molecule cannot fit in one periodic image")
    ads_place.add_argument("--fit-padding", type=float, default=0.15, help="fractional in-plane padding used with --auto-repeat-substrate")
    ads_place.add_argument("--site-type", required=True, choices=["top", "bridge", "hcp", "hcp_hollow", "fcc", "fcc_hollow", "hollow", "fourfold_hollow"])
    ads_place.add_argument("--site-index", type=int, default=1)
    ads_place.add_argument("--height", type=float, default=2.5, help="height above the top layer in angstrom")
    ads_place.add_argument("--rotate", type=float, default=0.0, help="rotation about the c axis in degrees")
    ads_place.add_argument("--tilt", type=float, default=0.0, help="tilt/pitch angle in degrees")
    ads_place.add_argument("--roll", type=float, default=0.0, help="roll angle in degrees")
    ads_place.add_argument(
        "--keep-cell-height",
        action="store_true",
        help="write the substrate cell unchanged instead of lengthening c to keep its vacuum gap",
    )

    ads_move = adsorbate_sub.add_parser("move", formatter_class=HelpFormatter, help="move a top-side molecule in a stacked structure")
    ads_move.add_argument("poscar_path")
    ads_move.add_argument("--target-cart", type=parse_float_vector, default=None)
    ads_move.add_argument("--target-direct", type=parse_float_vector, default=None)
    ads_move.add_argument("--rotate", type=float, default=0.0)
    ads_move.add_argument("--tilt", type=float, default=0.0)
    ads_move.add_argument("--roll", type=float, default=0.0)
    ads_move.add_argument(
        "--keep-cell-height",
        action="store_true",
        help="write the cell unchanged instead of lengthening c to keep its vacuum gap",
    )

    ads_assemble = adsorbate_sub.add_parser(
        "assemble",
        formatter_class=HelpFormatter,
        help="match a substrate cell to an experimental molecular assembly lattice",
        description=(
            "Advanced search mode: build a synthetic molecular lattice from a, b, and angle, then find "
            "substrate supercells that can support that periodic assembly. This does not place a molecule."
        ),
    )
    ads_assemble.add_argument("substrate_poscar")
    ads_assemble.add_argument("--a-length", type=float, required=True, help="target a length in angstrom")
    ads_assemble.add_argument("--b-length", type=float, default=None, help="target b length in angstrom; defaults to a")
    ads_assemble.add_argument("--angle", type=float, default=60.0, help="target in-plane angle in degrees")
    ads_assemble.add_argument("--length", dest="max_length", type=parse_positive_float, required=True, help="maximum in-plane supercell length in angstrom")
    ads_assemble.add_argument("--top-strain", type=parse_nonnegative_float, required=True, help="target molecular-lattice principal logarithmic strain budget as a fraction")
    ads_assemble.add_argument("--bottom-strain", type=parse_nonnegative_float, required=True, help="substrate principal logarithmic strain budget as a fraction")
    ads_assemble.add_argument("--preview-limit", type=int, default=10, help="number of lowest-strain candidates to print after the search; use 0 to hide")

    ads_path = adsorbate_sub.add_parser(
        "path",
        formatter_class=HelpFormatter,
        help="build the chain of images between two structures of one cell",
        description=(
            "Write the evenly spaced chain of images a nudged-elastic-band run starts from -- a "
            "molecule or adatom diffusing from one site to the next, say. 00/POSCAR is the "
            "initial structure and the last folder the final one. The atoms of the two endpoints "
            "are paired by the assignment that makes the path shortest, and every atom travels to "
            "its nearest periodic image. Both endpoints must share one cell and one composition."
        ),
    )
    ads_path.add_argument("start_structure")
    ads_path.add_argument("end_structure")
    ads_path.add_argument(
        "--images",
        type=int,
        default=3,
        help="number of intermediate images; the chain holds two more than this",
    )
    ads_path.add_argument(
        "--no-match",
        dest="match",
        action="store_false",
        help="pair the atoms in file order instead of by the shortest-path assignment",
    )
    ads_path.add_argument("--output-dir", default=None, help="where to write the numbered image folders")
