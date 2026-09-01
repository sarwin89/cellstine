"""Dependency-free parser group for symmetry commands."""

from __future__ import annotations

import argparse

from .argtypes import (
    HelpFormatter,
    parse_mesh_shift,
    parse_positive_float,
    parse_positive_int,
    parse_supercell,
)


def add_symmetry_group(groups: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the dependency-free ``symmetry`` command group."""

    symmetry = groups.add_parser(
        "symmetry",
        formatter_class=HelpFormatter,
        help="symmetry analysis and cell reduction",
        description="Symmetry workflows. The native engine computes symmetry operations, point groups, equivalent-atom orbits, primitive cells, and Niggli/Delaunay reductions; direct spglib adds space-group types and Wyckoff labels when installed.",
    )
    symmetry_sub = symmetry.add_subparsers(dest="stage")

    sym_analyse = symmetry_sub.add_parser(
        "analyse",
        formatter_class=HelpFormatter,
        help="analyse space group, operations, Wyckoff labels, and equivalent atoms",
    )
    sym_analyse.add_argument("structure")
    sym_analyse.add_argument("--backend", choices=["auto", "native", "spglib"], default="auto")
    sym_analyse.add_argument("--symprec", type=float, default=0.01, help="Cartesian distance tolerance for symmetry finding")
    sym_analyse.add_argument("--angle-tolerance", type=float, default=5.0, help="angle tolerance in degrees for symmetry finding")

    sym_reduce = symmetry_sub.add_parser(
        "reduce",
        formatter_class=HelpFormatter,
        help="write a primitive, conventional, or refined cell",
    )
    sym_reduce.add_argument("structure")
    sym_reduce.add_argument("--cell", choices=["primitive", "conventional", "refined"], default="primitive")
    sym_reduce.add_argument("--backend", choices=["auto", "native", "spglib"], default="auto")
    sym_reduce.add_argument("--symprec", type=float, default=0.01, help="Cartesian distance tolerance for symmetry finding")
    sym_reduce.add_argument("--angle-tolerance", type=float, default=5.0, help="angle tolerance in degrees for symmetry finding")
    sym_reduce.add_argument("--output", default=None, help="optional output POSCAR path")

    sym_lattice_reduce = symmetry_sub.add_parser(
        "lattice-reduce",
        formatter_class=HelpFormatter,
        help="write a Niggli- or Delaunay-reduced lattice representation",
    )
    sym_lattice_reduce.add_argument("structure")
    sym_lattice_reduce.add_argument("--reduction", choices=["niggli", "delaunay"], default="niggli")
    sym_lattice_reduce.add_argument("--backend", choices=["auto", "native", "spglib"], default="auto")
    sym_lattice_reduce.add_argument("--symprec", type=float, default=1e-5, help="lattice reduction tolerance")
    sym_lattice_reduce.add_argument("--output", default=None, help="optional output POSCAR path")

    sym_kpoints = symmetry_sub.add_parser(
        "kpoints",
        formatter_class=HelpFormatter,
        help="write a symmetry-reduced Brillouin-zone sampling mesh",
        description=(
            "Write a KPOINTS file for a structure. The mesh follows either a largest allowed "
            "reciprocal-space step (--spacing, the quantity VASP calls KSPACING, in 1/angstrom "
            "with the 2 pi convention) or explicit divisions, and is reduced by the rotations of "
            "the space group of the cell together with time reversal. The weights written are "
            "exact orbit sizes and add up to the size of the unreduced mesh."
        ),
    )
    sym_kpoints.add_argument("structure")
    sym_kpoints.add_argument(
        "--spacing",
        type=parse_positive_float,
        default=None,
        help="largest allowed step between sampled wavevectors, in 1/angstrom",
    )
    sym_kpoints.add_argument(
        "--divisions",
        type=parse_supercell,
        default=None,
        help="explicit mesh divisions as 'n1,n2,n3' instead of a spacing",
    )
    sym_kpoints.add_argument(
        "--mesh",
        choices=["gamma", "monkhorst"],
        default="gamma",
        help="Gamma-centred mesh, or the Monkhorst-Pack half-step offset on even axes",
    )
    sym_kpoints.add_argument(
        "--shift",
        type=parse_mesh_shift,
        default=None,
        help="explicit mesh offset in grid steps as 's1,s2,s3', overriding --mesh",
    )
    sym_kpoints.add_argument(
        "--surface",
        action="store_true",
        help="sample the surface normal with a single point, as a slab with vacuum needs",
    )
    sym_kpoints.add_argument("--no-symmetry", action="store_true", help="reduce by time reversal only")
    sym_kpoints.add_argument("--no-time-reversal", action="store_true", help="do not identify k with -k")
    sym_kpoints.add_argument("--list-points", action="store_true", help="always write the explicit irreducible list with weights")
    sym_kpoints.add_argument("--automatic", action="store_true", help="always write the automatic mesh line instead of the list")
    sym_kpoints.add_argument("--symprec", type=float, default=0.01, help="Cartesian distance tolerance for symmetry finding")
    sym_kpoints.add_argument("--output", default=None, help="optional output KPOINTS path")

    sym_kpath = symmetry_sub.add_parser(
        "kpath",
        formatter_class=HelpFormatter,
        help="write a band-structure path through the Brillouin zone",
        description=(
            "Write the line-mode KPOINTS file of a band structure. The high-symmetry points are "
            "derived from the symmetry of the cell -- they are the points and the ends of the "
            "symmetry lines of the Brillouin zone, not a table look-up -- and are named after the "
            "conventional cell of the Bravais lattice. The walk is the conventional one for the "
            "Bravais types that have one and is otherwise derived from the symmetry lines; --path "
            "chooses it explicitly, as 'GAMMA-X-W|K-L'."
        ),
    )
    sym_kpath.add_argument("structure")
    sym_kpath.add_argument(
        "--spacing",
        type=parse_positive_float,
        default=None,
        help="largest allowed step along the path, in 1/angstrom (default 0.03)",
    )
    sym_kpath.add_argument(
        "--divisions",
        type=parse_positive_int,
        default=None,
        help="explicit number of points per segment instead of a spacing",
    )
    sym_kpath.add_argument(
        "--path",
        default=None,
        help="explicit walk, such as 'GAMMA-X-W|K-L'; '|' breaks the line",
    )
    sym_kpath.add_argument(
        "--derived-path",
        action="store_true",
        help="always derive the walk from the symmetry lines, never use the conventional one",
    )
    sym_kpath.add_argument("--no-symmetry", action="store_true", help="use the point group of the lattice, ignoring the atoms")
    sym_kpath.add_argument("--no-time-reversal", action="store_true", help="do not identify k with -k")
    sym_kpath.add_argument("--symprec", type=float, default=0.01, help="Cartesian distance tolerance for symmetry finding")
    sym_kpath.add_argument("--output", default=None, help="optional output KPOINTS path")
