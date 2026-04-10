"""Argument parser construction for the grouped CELLSTINE CLI."""

from __future__ import annotations

import argparse
from typing import List


APP_NAME = "CELLSTINE"
APP_EXPANSION = "CELL Superlattice Transformation INterface and Engine"


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Readable help text with examples and defaults."""


def parse_index_spec(raw: str) -> List[int]:
    values: List[int] = []
    for chunk in str(raw).split(","):
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
        raise argparse.ArgumentTypeError("please provide at least one index")
    return list(dict.fromkeys(values))


def parse_float_vector(raw: str) -> List[float]:
    values = [float(token.strip()) for token in str(raw).replace(";", ",").split(",") if token.strip()]
    if len(values) not in {2, 3}:
        raise argparse.ArgumentTypeError("please provide 2 or 3 numeric values separated by commas")
    return values


def parse_int_matrix(raw: str) -> List[int]:
    values = [int(token.strip()) for token in str(raw).replace(";", ",").split(",") if token.strip()]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("please provide exactly four integer values")
    return values


def parse_optional_float_list(raw: str | None) -> List[float] | None:
    if raw in {None, ""}:
        return None
    return [float(token.strip()) for token in str(raw).replace(";", ",").split(",") if token.strip()]


def parse_prestrain_modes(raw: str | None) -> List[str] | None:
    if raw in {None, ""}:
        return None
    return [token.strip().lower() for token in str(raw).split(",") if token.strip()]


def parse_string_list(raw: str | None) -> List[str] | None:
    if raw in {None, ""}:
        return None
    return [token.strip() for token in str(raw).split(",") if token.strip()]


def parse_angles_by_layer(raw: str | None) -> List[List[float] | None] | None:
    if raw in {None, ""}:
        return None
    groups = []
    for chunk in str(raw).split(";"):
        token = chunk.strip()
        if not token:
            groups.append(None)
        else:
            groups.append([float(item.strip()) for item in token.split(",") if item.strip()])
    return groups


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cellstine",
        formatter_class=HelpFormatter,
        description=(
            f"{APP_NAME} | {APP_EXPANSION}\n\n"
            "Three grouped workflows are available:\n"
            "  moire      Build commensurate moire supercells\n"
            "  adsorbate  Place and manipulate molecules on substrates\n"
            "  interface  Build surfaces and heterointerfaces from bulk or slab inputs"
        ),
    )
    parser.add_argument("--version", action="store_true", help="show package and optional dependency versions")
    groups = parser.add_subparsers(dest="group")

    moire = groups.add_parser("moire", formatter_class=HelpFormatter, help="moire supercell construction")
    moire_sub = moire.add_subparsers(dest="stage")

    moire_find = moire_sub.add_parser(
        "find",
        formatter_class=HelpFormatter,
        help="search bilayer commensurate candidates",
        description="Search bilayer commensurate candidates and save the results table for later generation.",
    )
    moire_find.add_argument("top_poscar")
    moire_find.add_argument("bottom_poscar")
    moire_find.add_argument("--nindex", type=int, default=12, help="maximum supercell index search size")
    moire_find.add_argument("--min-angle", type=float, default=0.0, help="minimum twist angle in degrees")
    moire_find.add_argument("--max-angle", type=float, default=None, help="maximum twist angle in degrees")
    moire_find.add_argument("--angle-step", type=float, default=0.1, help="fallback angle scan spacing in degrees")
    moire_find.add_argument("--angles", type=parse_optional_float_list, default=None, help="comma-separated explicit twist angles in degrees")
    moire_find.add_argument("--vector-tolerance", type=float, default=2e-3, help="vector mismatch tolerance as a fraction")
    moire_find.add_argument("--vector-strain-tolerance", type=float, default=2e-3, help="vector strain tolerance as a fraction")
    moire_find.add_argument("--candidate-tolerance", type=float, default=None, help="candidate merge tolerance as a fraction")
    moire_find.add_argument("--strain-tolerance", type=float, default=None, help="strain tolerance as a fraction")
    moire_find.add_argument("--matrix-values", type=parse_int_matrix, default=None, help="comma-separated 2x2 matrix entries in any order")
    moire_find.add_argument("--matrix-layer", choices=["1", "2", "either"], default="either")
    moire_find.add_argument("--matrix-match-mode", choices=["absolute", "exact"], default="absolute")
    moire_find.add_argument("--workers", type=int, default=1, help="process workers for angle-parallel search")
    moire_find.add_argument("--top-c-repeat", type=int, default=1)
    moire_find.add_argument("--bottom-c-repeat", type=int, default=1)
    moire_find.add_argument("--prestrain-top-mode", choices=["none", "biaxial", "uniaxial"], default="none")
    moire_find.add_argument("--prestrain-top-value", type=float, default=0.0)
    moire_find.add_argument("--prestrain-top-axis", default=None)
    moire_find.add_argument("--prestrain-bottom-mode", choices=["none", "biaxial", "uniaxial"], default="none")
    moire_find.add_argument("--prestrain-bottom-value", type=float, default=0.0)
    moire_find.add_argument("--prestrain-bottom-axis", default=None)

    moire_findn = moire_sub.add_parser("findn", formatter_class=HelpFormatter, help="search N-layer commensuration candidates")
    moire_findn.add_argument("bottom_poscar")
    moire_findn.add_argument("upper_poscars", nargs="+")
    moire_findn.add_argument("--match-mode", choices=["base_shared", "base_independent", "pairwise"], default="base_shared")
    moire_findn.add_argument("--nindex", type=int, default=12)
    moire_findn.add_argument("--min-angles", type=parse_optional_float_list, default=None, help="comma-separated minimum angles for each upper layer")
    moire_findn.add_argument("--max-angles", type=parse_optional_float_list, default=None, help="comma-separated maximum angles for each upper layer")
    moire_findn.add_argument("--angles-by-layer", type=parse_angles_by_layer, default=None, help="semicolon-separated explicit angle lists per upper layer")
    moire_findn.add_argument("--vector-tolerance", type=float, default=2e-3)
    moire_findn.add_argument("--vector-strain-tolerance", type=float, default=2e-3)
    moire_findn.add_argument("--candidate-tolerance", type=float, default=None)
    moire_findn.add_argument("--max-atoms", type=int, default=2000)
    moire_findn.add_argument("--workers", type=int, default=1)
    moire_findn.add_argument("--bottom-c-repeat", type=int, default=1)
    moire_findn.add_argument("--upper-c-repeats", type=parse_optional_float_list, default=None, help="comma-separated c repeats for upper layers")
    moire_findn.add_argument("--prestrain-modes", type=parse_prestrain_modes, default=None, help="comma-separated prestrain modes for bottom then upper layers")
    moire_findn.add_argument("--prestrain-values", type=parse_optional_float_list, default=None, help="comma-separated prestrain magnitudes for bottom then upper layers")
    moire_findn.add_argument("--prestrain-axes", type=parse_string_list, default=None, help="comma-separated strain axes for bottom then upper layers")

    moire_make = moire_sub.add_parser("make", formatter_class=HelpFormatter, help="generate bilayer supercells from saved results")
    moire_make.add_argument("results_file")
    moire_make.add_argument("--indexes", type=parse_index_spec, required=True, help="comma-separated indices or ranges, e.g. 1,3-5")
    moire_make.add_argument("--interlayer-distance", type=float, default=3.35, help="layer separation in angstrom")
    moire_make.add_argument("--workers", type=int, default=1)
    moire_make.add_argument("--output-dir", default=None)

    moire_maken = moire_sub.add_parser("maken", formatter_class=HelpFormatter, help="generate N-layer supercells from saved results")
    moire_maken.add_argument("results_file")
    moire_maken.add_argument("--indexes", type=parse_index_spec, required=True)
    moire_maken.add_argument("--interlayers", type=parse_optional_float_list, required=True, help="comma-separated interlayer distances in angstrom")
    moire_maken.add_argument("--output-dir", default=None)

    moire_translate = moire_sub.add_parser("translate", formatter_class=HelpFormatter, help="translate the upper layer in a stacked bilayer")
    moire_translate.add_argument("poscar_path")
    moire_translate.add_argument("--shift-cart", type=parse_float_vector, default=None, help="cartesian shift vector in angstrom")
    moire_translate.add_argument("--shift-direct", type=parse_float_vector, default=None, help="direct shift vector")

    moire_translaten = moire_sub.add_parser("translaten", formatter_class=HelpFormatter, help="translate the uppermost layer in a stacked N-layer structure")
    moire_translaten.add_argument("poscar_path")
    moire_translaten.add_argument("--shift-cart", type=parse_float_vector, default=None)
    moire_translaten.add_argument("--shift-direct", type=parse_float_vector, default=None)

    moire_visualize = moire_sub.add_parser("visualize", formatter_class=HelpFormatter, help="build a moire visualization HTML")
    moire_visualize.add_argument("results_file")
    moire_visualize.add_argument("--indices", type=parse_index_spec, default=None)
    moire_visualize.add_argument("--interlayer", type=float, default=3.35)

    adsorbate = groups.add_parser("adsorbate", formatter_class=HelpFormatter, help="molecule on substrate workflows")
    adsorbate_sub = adsorbate.add_subparsers(dest="stage")

    ads_place = adsorbate_sub.add_parser("place", formatter_class=HelpFormatter, help="place a molecule on a substrate site")
    ads_place.add_argument("substrate_poscar")
    ads_place.add_argument("molecule_poscar")
    ads_place.add_argument("--substrate-kind", choices=["bulk", "substrate", "patch", "surface", "slab"], default="substrate")
    ads_place.add_argument("--miller", default="1,1,1", help="surface Miller indices for bulk inputs")
    ads_place.add_argument("--layers", type=int, default=4)
    ads_place.add_argument("--vacuum", type=float, default=15.0)
    ads_place.add_argument("--site-type", required=True, choices=["top", "bridge", "hcp", "fcc", "hollow", "fourfold_hollow"])
    ads_place.add_argument("--site-index", type=int, default=1)
    ads_place.add_argument("--height", type=float, default=2.5, help="height above the top layer in angstrom")
    ads_place.add_argument("--rotate", type=float, default=0.0, help="rotation about the c axis in degrees")

    ads_move = adsorbate_sub.add_parser("move", formatter_class=HelpFormatter, help="move a top-side molecule in a stacked structure")
    ads_move.add_argument("poscar_path")
    ads_move.add_argument("--target-cart", type=parse_float_vector, default=None)
    ads_move.add_argument("--target-direct", type=parse_float_vector, default=None)
    ads_move.add_argument("--rotate", type=float, default=0.0)

    ads_assemble = adsorbate_sub.add_parser("assemble", formatter_class=HelpFormatter, help="find a commensurate substrate cell for a molecular assembly lattice")
    ads_assemble.add_argument("substrate_poscar")
    ads_assemble.add_argument("--a-length", type=float, required=True, help="target a length in angstrom")
    ads_assemble.add_argument("--b-length", type=float, default=None, help="target b length in angstrom; defaults to a")
    ads_assemble.add_argument("--angle", type=float, default=60.0, help="target in-plane angle in degrees")
    ads_assemble.add_argument("--nindex", type=int, default=12)
    ads_assemble.add_argument("--max-strain", type=float, default=0.05, help="maximum allowed strain as a fraction")

    ads_visualize = adsorbate_sub.add_parser("visualize", formatter_class=HelpFormatter, help="visualize a slab or adsorbate POSCAR")
    ads_visualize.add_argument("structure_path")

    interface = groups.add_parser("interface", formatter_class=HelpFormatter, help="surface and heterointerface workflows")
    interface_sub = interface.add_subparsers(dest="stage")

    int_surface = interface_sub.add_parser("surface", formatter_class=HelpFormatter, help="build a slab from a bulk structure")
    int_surface.add_argument("bulk_poscar")
    int_surface.add_argument("--miller", required=True, help="Miller indices like 1,1,1 or 1,1,2x")
    int_surface.add_argument("--layers", type=int, default=4)
    int_surface.add_argument("--vacuum", type=float, default=15.0)
    int_surface.add_argument("--repeat-a", type=int, default=1)
    int_surface.add_argument("--repeat-b", type=int, default=1)
    int_surface.add_argument("--supercell-matrix", type=parse_int_matrix, default=None)
    int_surface.add_argument("--analyse-sites", action="store_true")

    int_sites = interface_sub.add_parser("sites", formatter_class=HelpFormatter, help="identify adsorption sites for a slab")
    int_sites.add_argument("slab_poscar")
    int_sites.add_argument("--surface-side", choices=["top", "bottom"], default="top")

    int_build = interface_sub.add_parser("build", formatter_class=HelpFormatter, help="build a heterointerface from slabs or bulks")
    int_build.add_argument("bottom_input")
    int_build.add_argument("top_input")
    int_build.add_argument("--bottom-kind", choices=["bulk", "surface", "slab"], default="surface")
    int_build.add_argument("--top-kind", choices=["bulk", "surface", "slab"], default="surface")
    int_build.add_argument("--bottom-miller", default="1,1,1")
    int_build.add_argument("--top-miller", default="1,1,1")
    int_build.add_argument("--bottom-layers", type=int, default=4)
    int_build.add_argument("--top-layers", type=int, default=4)
    int_build.add_argument("--bottom-vacuum", type=float, default=15.0)
    int_build.add_argument("--top-vacuum", type=float, default=15.0)
    int_build.add_argument("--gap", type=float, default=3.0)

    int_match = interface_sub.add_parser("match", formatter_class=HelpFormatter, help="scan bulk surfaces for interface matches")
    int_match.add_argument("bottom_bulk")
    int_match.add_argument("top_bulk")
    int_match.add_argument("--bottom-millers", nargs="*", default=None, help="list of Miller indices such as 1,0,0 1,1,1")
    int_match.add_argument("--top-millers", nargs="*", default=None, help="list of Miller indices such as 1,0,0 1,1,1")
    int_match.add_argument("--bottom-layers-list", nargs="*", type=int, default=None, help="candidate bottom slab layer counts")
    int_match.add_argument("--top-layers-list", nargs="*", type=int, default=None, help="candidate top slab layer counts")
    int_match.add_argument("--vacuum", type=float, default=15.0)
    int_match.add_argument("--max-strain", type=float, default=0.05)

    int_visualize = interface_sub.add_parser("visualize", formatter_class=HelpFormatter, help="visualize a slab or interface POSCAR")
    int_visualize.add_argument("structure_path")

    return parser
