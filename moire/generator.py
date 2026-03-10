"""Construct a combined supercell from a results table.

The output is a POSCAR file containing atoms from both layers placed in a
common superlattice.  The generator reads the file produced by
``moire.finder`` (or the legacy scripts) and uses the integer coefficient
information to compute the supercell vectors.

Only a basic replication algorithm is implemented; there is no attempt
to remove overlapping atoms other than simple rounding.  Advanced
features such as z-fixing and selective dynamics are intentionally
omitted to keep the core logic small and easy to reuse.
"""

import numpy as np
import argparse
from typing import Tuple, List

from . import io as io_mod


def parse_results(filename: str) -> Tuple[str, str, List[dict]]:
    """Parse a results file.

    Returns a tuple ``(file1, file2, records)``.  ``records`` is a list of
    dictionaries containing the parsed values for each row.  The first
    line of the file is expected to contain the two input filenames.
    """
    records = []
    with open(filename, 'r') as f:
        first = f.readline().strip().split()
        if len(first) < 2:
            raise ValueError("results file does not contain input filenames")
        file1, file2 = first[0], first[1]
        # skip header lines until a line beginning with '| idx' is found
        for line in f:
            if line.lstrip().startswith('| idx'):
                break
        # next line is separator; skip it
        next(f, None)
        # now parse data rows
        for line in f:
            if not line.strip() or line.strip().startswith('-'):
                continue
            parts = [p.strip() for p in line.strip().split('|') if p.strip()]
            if len(parts) < 12:
                continue
            idx = int(parts[0])
            angle = float(parts[1])
            strain_avg = float(parts[2])
            strain1 = float(parts[3])
            strain2 = float(parts[4])
            atoms = int(parts[5])
            ratio = parts[6]
            # coefficients are in parts 7..10 each containing two ints
            i11, i12 = [int(x) for x in parts[7].split()]
            i21, i22 = [int(x) for x in parts[8].split()]
            j11, j12 = [int(x) for x in parts[9].split()]
            j21, j22 = [int(x) for x in parts[10].split()]
            eps1 = float(parts[11]) if len(parts) > 11 else 0.0
            eps2 = float(parts[12]) if len(parts) > 12 else 0.0
            records.append({
                'idx': idx,
                'angle': angle,
                'strain_avg': strain_avg,
                'strain1': strain1,
                'strain2': strain2,
                'atoms': atoms,
                'ratio': ratio,
                'i11': i11, 'i12': i12, 'i21': i21, 'i22': i22,
                'j11': j11, 'j12': j12, 'j21': j21, 'j22': j22,
                'eps1': eps1, 'eps2': eps2,
            })
    return file1, file2, records


def build_supercell(pos1: str,
                    pos2: str,
                    coef: dict,
                    shift1: Tuple[float,float] = (0.0,0.0),
                    shift2: Tuple[float,float] = (0.0,0.0)) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    """Generate supercell lattice and atomic positions given coefficients.

    Returns ``(lattice, positions, counts, types)`` suitable for
    passing to ``io.write_poscar``.
    """
    # read both crystals
    lat1, coords1, counts1, types1 = io_mod.parse_poscar(pos1)
    lat2, coords2, counts2, types2 = io_mod.parse_poscar(pos2)
    # compute supercell basis from first layer coefficients
    i11, i12, i21, i22 = coef['i11'], coef['i12'], coef['i21'], coef['i22']
    j11, j12, j21, j22 = coef['j11'], coef['j12'], coef['j21'], coef['j22']
    # assemble two in-plane super vectors for each layer
    v1 = i11 * lat1[0] + i12 * lat1[1]
    v2 = i21 * lat1[0] + i22 * lat1[1]
    g1 = j11 * lat2[0] + j12 * lat2[1]
    g2 = j21 * lat2[0] + j22 * lat2[1]
    # choose final in-plane vectors as average of the two (simple compromise)
    s1 = 0.5 * (v1 + g1)
    s2 = 0.5 * (v2 + g2)
    # choose z vector to be the larger of the two input c vectors
    zlen = max(np.linalg.norm(lat1[2]), np.linalg.norm(lat2[2]))
    s3 = np.array([0.0, 0.0, zlen])
    super_lat = np.vstack((s1, s2, s3))
    inv_super = np.linalg.inv(super_lat)

    # helper to convert and wrap coordinates into supercell
    def wrap(coords: np.ndarray, shift: Tuple[float,float]) -> List[Tuple[float]]:
        # apply xy shift in direct supercell coords
        cart = coords.copy()
        cart[:,0] += shift[0]; cart[:,1] += shift[1]
        direct = cart @ inv_super
        wrapped = direct - np.floor(direct)
        return wrapped @ super_lat

    pos_list = []
    type_list = []

    # layer 1 atoms
    wrapped1 = wrap(coords1, shift1)
    for p in wrapped1:
        pos_list.append(p)
    if types1:
        for ct, count in zip(types1, counts1):
            type_list += [ct] * count
    else:
        type_list += ['X'] * len(wrapped1)

    # layer 2 atoms
    wrapped2 = wrap(coords2, shift2)
    for p in wrapped2:
        pos_list.append(p)
    if types2:
        for ct, count in zip(types2, counts2):
            type_list += [ct] * count
    else:
        type_list += ['Y'] * len(wrapped2)

    # compute counts from type_list
    unique_types = []
    counts = []
    for t in type_list:
        if t in unique_types:
            counts[unique_types.index(t)] += 1
        else:
            unique_types.append(t)
            counts.append(1)

    positions = np.array(pos_list)
    return super_lat, positions, counts, unique_types


def main():
    p = argparse.ArgumentParser(description="Generate a merged supercell from a finder results file")
    p.add_argument('results', help='file containing finder results')
    p.add_argument('index', type=int, help='1-based index of the candidate to build')
    p.add_argument('--shift1', nargs=2, type=float, default=(0.0,0.0),
                   help='in-plane shift applied to layer1 atoms (fractional)')
    p.add_argument('--shift2', nargs=2, type=float, default=(0.0,0.0),
                   help='in-plane shift applied to layer2 atoms (fractional)')
    p.add_argument('--output', default='supercell.vasp',
                   help='name of POSCAR to write')
    args = p.parse_args()

    file1, file2, records = parse_results(args.results)
    recs_by_idx = {r['idx']: r for r in records}
    if args.index not in recs_by_idx:
        raise ValueError(f"Index {args.index} not found in {args.results}")
    coef = recs_by_idx[args.index]
    lat, positions, counts, types = build_supercell(file1, file2, coef,
                                                   tuple(args.shift1), tuple(args.shift2))
    io_mod.write_poscar(args.output, lat, positions, counts, types,
                        comment=f"supercell from {args.results} idx {args.index}")
    print(f"Wrote supercell to {args.output}")

if __name__ == '__main__':
    main()
