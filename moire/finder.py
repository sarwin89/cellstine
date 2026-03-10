"""Search for commensurate supercell angles between two lattices.

This module exposes both a programmatic API and a CLI.  The ``main``
function can be installed as an entry point for a command such as
``moire-find`` or ``finder``; for now the top-level script ``finder.py``
invokes ``moire.finder.main()``.
"""

import numpy as np
import argparse
from typing import List, Tuple

from . import io as io_mod
from . import lattice as lat


def find_supercells(lat1: np.ndarray,
                    lat2: np.ndarray,
                    angle_lower: float,
                    angle_upper: float,
                    angle_step: float = 0.001,
                    nindex: int = 10,
                    tol: float = 1e-5,
                    lin_tol: float = 1e-4,
                    strain_tol: float = None,
                    strain_layer: str = 'avg',
                    min_atoms: int = None,
                    max_atoms: int = None) -> List[Tuple]:
    """Return a list of candidate supercells.

    Each entry is a tuple matching the format produced previously by
    the old ``cellmatch2``/``cellfind`` code.  See the documentation in
    the CLI for a description of the tuple elements.
    """
    cnt1 = cnt2 = None  # counts are not used during the geometric scan

    angles = np.arange(angle_lower, angle_upper + angle_step, angle_step)
    tasks = []
    for ang in angles:
        tasks.append((ang, lat1, lat2, nindex, tol, lin_tol, 1, 1))
    # simple serial loop for now; multiprocessing can be added later

    all_results = []
    for ang in angles:
        theta = np.radians(ang)
        rot1 = np.array([lat.rotate_vector(v, theta) for v in lat1[:2, :2]])
        pairs = lat.gen_pairs(rot1, lat2[:2, :2], nindex, tol)
        sc = lat.build_supercells(pairs, lin_tol, rot1, lat2[:2, :2], 1, 1, ang)
        all_results.append(sc)

    flat = [item for sub in all_results for item in sub]
    # apply filters like in the CLI
    filtered = []
    for rec in flat:
        strain_avg, s1, s2, om1, om2, atoms, *rest = rec
        if min_atoms is not None and atoms < min_atoms:
            continue
        if max_atoms is not None and atoms > max_atoms:
            continue
        if strain_tol is not None:
            if strain_layer == 'avg' and strain_avg > strain_tol:
                continue
            if strain_layer == '1' and s1 > strain_tol:
                continue
            if strain_layer == '2' and s2 > strain_tol:
                continue
        filtered.append(rec)
    # uniqueness filtering (keep smallest atoms)
    uniq = {}
    for rec in filtered:
        strain_avg, s1, s2, om1, om2, atoms, *rest = rec
        key = (round(strain_avg / lin_tol), om1, om2)
        if key not in uniq or atoms < uniq[key][5]:
            uniq[key] = rec
    return sorted(uniq.values(), key=lambda x: x[0])


# geometry helpers are provided by the lattice module; nothing to
# duplicate here.


# Command-line interface

def parse_args():
    p = argparse.ArgumentParser(description="Finder for commensurate supercells")
    p.add_argument('pos1'); p.add_argument('pos2')
    p.add_argument('angle_lower', type=float, nargs='?', default=None,
                   help='lower bound for rotation angle')
    p.add_argument('angle_upper', type=float, nargs='?', default=None,
                   help='upper bound for rotation angle')
    p.add_argument('--angle_step', type=float, default=0.001,
                   help='step size when scanning a range')
    p.add_argument('--angles', type=str, default=None,
                   help='comma-separated list of discrete angles to test')
    p.add_argument('--nindex', type=int, default=10)
    p.add_argument('--tolerance', type=float, default=0.01,
                   help='linear matching tolerance for basis vectors')
    p.add_argument('--lin_tol', type=float, default=0.02,
                   help='max relative error used when building supercells')
    p.add_argument('--strain_tol', type=float, default=None,
                   help='reject candidates with strain above this value')
    p.add_argument('--strain_layer', choices=['avg','1','2'], default='avg',
                   help='which strain metric to use when applying --strain_tol')
    p.add_argument('--max_atoms', type=int, default=None,
                   help='maximum number of atoms in the combined supercell')
    p.add_argument('--min_atoms', type=int, default=None,
                   help='minimum number of atoms in the combined supercell')
    p.add_argument('--processes', type=int, default=4)
    p.add_argument('--output', type=str, default='results.dat')
    return p.parse_args()


def main():
    args = parse_args()
    lat1, _, _, _ = io_mod.parse_poscar(args.pos1)
    lat2, _, _, _ = io_mod.parse_poscar(args.pos2)
    # determine angle list
    if args.angles:
        # explicit list overrides numeric bounds
        angle_list = [float(a) for a in args.angles.split(',')]
        angle_lower = min(angle_list)
        angle_upper = max(angle_list)
        step = None
    else:
        if args.angle_lower is None or args.angle_upper is None:
            raise ValueError('must specify either angle range or --angles list')
        angle_list = None
        angle_lower = args.angle_lower
        angle_upper = args.angle_upper
        step = args.angle_step

    results = find_supercells(lat1, lat2,
                               angle_lower, angle_upper,
                               step or 0.001, args.nindex,
                               args.tolerance, args.lin_tol,
                               args.strain_tol, args.strain_layer,
                               args.min_atoms, args.max_atoms)
    # write results to file and stdout
    with open(args.output, 'w') as f:
        # store the input file names on the first line so the generator
        # script can automatically pick them up later
        f.write(f"{args.pos1} {args.pos2}\n")
        header = (
            "| idx | angle (deg) | strain_avg | strain1 | strain2 | atoms | ratio "
            "| i11 i12 | i21 i22 | j11 j12 | j21 j22 | eps1 | eps2 |\n"
        )
        sep = '-' * len(header) + '\n'
        f.write(header); f.write(sep)
        for i, rec in enumerate(results, 1):
            strain_avg, s1, s2, om1, om2, atoms, c1p, c1q, c2p, c2q, eps1, eps2, ang = rec
            i11, i12 = c1p; i21, i22 = c1q
            j11, j12 = c2p; j21, j22 = c2q
            f.write(
                f"|{i:4d} | {ang:10.4f} | {strain_avg:10.6f} | {s1:8.6f} | {s2:8.6f} | "
                f"{atoms:5d} | {om1:3d}/{om2:<3d} | {i11:4d} {i12:4d} | {i21:4d} {i22:4d} | "
                f"{j11:4d} {j12:4d} | {j21:4d} {j22:4d} | {eps1:8.2e} | {eps2:8.2e} |\n"
            )
    print("Results written to", args.output)
    if results:
        best = results[0]
        print(f"{len(results)} candidates found; best strain {best[0]:.6f} at angle {best[-1]:.4f}\n")
        # print top 5 for quick glance
        for idx, rec in enumerate(results[:5], 1):
            print(f"{idx:2d}: angle={rec[-1]:7.4f}°, strain_avg={rec[0]:.6g}, atoms={rec[5]}")
    else:
        print("no candidates found")
    for line in open(args.output):
        print(line, end='')


if __name__ == '__main__':
    main()
