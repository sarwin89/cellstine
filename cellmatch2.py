#!/usr/bin/env python3
import numpy as np
import argparse
import math
import multiprocessing as mp

# --- Utilities ---

def rotate_vector_2d(v, theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([c*v[0] - s*v[1], s*v[0] + c*v[1]])

def unit_area(v1, v2):
    return abs(v1[0]*v2[1] - v1[1]*v2[0])

def parse_poscar(path):
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    scale = float(lines[1].split()[0])
    lat = np.array([list(map(float, lines[i].split())) for i in (2,3,4)]) * scale
    counts = list(map(int, lines[6].split()))
    nats = sum(counts)
    coords = np.array([list(map(float, lines[i].split()[:3])) for i in range(8,8+nats)])
    return lat, coords, counts

# --- Candidate Pair Generation ---

def gen_pairs(rot1, lat2_2d, nidx, tol):
    rot = np.asarray(rot1)[:2,:2]
    lat2 = np.asarray(lat2_2d)[:2,:2]
    idx = np.arange(-nidx, nidx+1)
    c1 = np.vstack(np.meshgrid(idx,idx)).T.reshape(-1,2)
    c1 = c1[~np.all(c1==0, axis=1)]
    c2 = np.vstack(np.meshgrid(idx,idx)).T.reshape(-1,2)
    c2 = c2[~np.all(c2==0, axis=1)]

    V = c1 @ rot
    G = c2 @ lat2
    nV = np.linalg.norm(V, axis=1)
    nG = np.linalg.norm(G, axis=1)
    D = V[:,None,:] - G[None,:,:]
    rel_err = np.linalg.norm(D,axis=2)/(nV[:,None]+nG[None,:])

    pairs = []
    ai, bi = np.nonzero(rel_err < tol)
    for ia, ib in zip(ai, bi):
        pairs.append({
            'c1': tuple(c1[ia]),
            'c2': tuple(c2[ib]),
            'v1': tuple(V[ia]),
            'v2': tuple(G[ib]),
            'eps': float(rel_err[ia,ib])
        })
    pairs.sort(key=lambda p: p['eps'])
    return pairs

# --- Strain Calculation ---

def calc_strain(a1, b1, c1, a2, b2, c2):
    # enforce planar z=1
    a1x,a1y,a1z = a1; b1x,b1y,b1z = b1; c1z = 1.0
    a2x,a2y,a2z = a2; b2x,b2y,b2z = b2; c2z = 1.0
    M1 = np.array([
        [a1x*a1x+a1y*a1y+a1z*a1z, a1x*b1x+a1y*b1y+a1z*b1z, a1x*c1[0]+a1y*c1[1]+a1z*c1z],
        [a1x*b1x+a1y*b1y+a1z*b1z, b1x*b1x+b1y*b1y+b1z*b1z, b1x*c1[0]+b1y*c1[1]+b1z*c1z],
        [a1x*c1[0]+a1y*c1[1]+a1z*c1z, c1[0]*b1x+c1[1]*b1y+c1z*b1z, c1[0]*c1[0]+c1[1]*c1[1]+c1z*c1z]
    ])
    M2 = np.array([
        [a2x*a2x+a2y*a2y+a2z*a2z, a2x*b2x+a2y*b2y+a2z*b2z, a2x*c2[0]+a2y*c2[1]+a2z*c2z],
        [a2x*b2x+a2y*b2y+a2z*b2z, b2x*b2x+b2y*b2y+b2z*b2z, b2x*c2[0]+b2y*c2[1]+b2z*c2z],
        [a2x*c2[0]+a2y*c2[1]+a2z*c2z, c2[0]*b2x+c2[1]*b2y+c2z*b2z, c2[0]*c2[0]+c2[1]*c2[1]+c2z*c2z]
    ])
    rt1 = np.linalg.cholesky(M1).T
    rt2 = np.linalg.cholesky(M2).T
    E = rt2 @ np.linalg.inv(rt1) - np.eye(3)
    S = 0.5*(E + E.T + E @ E.T)
    ev = np.linalg.eigvals(S)
    return math.sqrt((ev*ev).sum())/3.0

# --- Build Supercells with 4 indices ---

def build_supercells(pairs, lin_tol, rot1, lat2, cnt1, cnt2):
    A1 = unit_area(rot1[0], rot1[1])
    A2 = unit_area(lat2[0], lat2[1])
    eps_arr = np.array([p['eps'] for p in pairs])
    k = (eps_arr <= lin_tol).sum()
    if k < 2:
        return []
    sub = pairs[:k]
    results = []
    for p in range(k):
        for q in range(p, k):
            c1p = sub[p]['c1']; c1q = sub[q]['c1']
            c2p = sub[p]['c2']; c2q = sub[q]['c2']
            v1 = np.array(sub[p]['v1']); v2 = np.array(sub[q]['v1'])
            g1 = np.array(sub[p]['v2']); g2 = np.array(sub[q]['v2'])
            surf1 = abs(v1[0]*v2[1] - v1[1]*v2[0])
            surf2 = abs(g1[0]*g2[1] - g1[1]*g2[0])
            if surf1 == 0 or surf2 == 0:
                continue
            om1 = round(surf1 / A1)
            om2 = round(surf2 / A2)
            if om1 <= 0 or om2 <= 0:
                continue
            strain = calc_strain([v1[0],v1[1],0], [v2[0],v2[1],0], [0,0,1],
                                 [g1[0],g1[1],0], [g2[0],g2[1],0], [0,0,1])
            atoms = round(cnt1 * om1 + cnt2 * om2)
            eps1 = sub[p]['eps']; eps2 = sub[q]['eps']
            results.append((strain, om1, om2, atoms, c1p, c1q, c2p, c2q, eps1, eps2))
    return sorted(results, key=lambda x: x[0])

# --- Angle Scan ---

def process_angle(args):
    angle, lat1, lat2, nidx, tol, lin_tol, cnt1, cnt2 = args
    theta = math.radians(angle)
    rot1 = np.array([rotate_vector_2d(v, theta) for v in lat1[:2]])
    pairs = gen_pairs(rot1, lat2[:2], nidx, tol)
    return build_supercells(pairs, lin_tol, rot1, lat2[:2], cnt1, cnt2)

# --- CLI ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pos1'); parser.add_argument('pos2')
    parser.add_argument('angle_lower', type=float); parser.add_argument('angle_upper', type=float)
    parser.add_argument('--angle_step', type=float, default=0.001)
    parser.add_argument('--nindex', type=int, default=10)
    parser.add_argument('--tolerance', type=float, default=0.01)
    parser.add_argument('--lin_tol', type=float, default=0.02)
    parser.add_argument('--processes', type=int, default=4)
    parser.add_argument('--output', type=str, default='results.dat')
    args = parser.parse_args()

    lat1, _, cnt1 = parse_poscar(args.pos1)
    lat2, _, cnt2 = parse_poscar(args.pos2)
    tot1, tot2 = sum(cnt1), sum(cnt2)

    angles = np.arange(
        args.angle_lower,
        args.angle_upper + args.angle_step,
        args.angle_step
    )
    # prepare arguments for each angle
    tasks = [
        (ang, lat1, lat2, args.nindex, args.tolerance, args.lin_tol, tot1, tot2)
        for ang in angles
    ]

    # run process_angle in parallel
    with mp.Pool(args.processes) as pool:
        all_results = pool.map(process_angle, tasks)

    # Debug: show how many candidates per angle
    for ang, subs in zip(angles, all_results):
        print(f"Angle {ang:.4f}° → {len(subs)} raw supercell candidates")

    flat = [item for sub in all_results for item in sub]
    print(f"Total candidates across all angles: {len(flat)}")

    # Unique by strain and atoms, keep smallest atoms
    uniq = {}
    for rec in flat:
        strain, om1, om2, atoms, *rest = rec
        key = (round(strain / args.lin_tol), om1, om2)
        if key not in uniq or atoms < uniq[key][3]:
            uniq[key] = rec
    results = sorted(uniq.values(), key=lambda x: x[0])

    # Output to file
    with open(args.output, 'w') as f:
        header = "| idx | strain   | atoms | ratio  | i11 i12 | i21 i22 | j11 j12 | j21 j22 | eps1     | eps2     |\n"
        sep = '-' * len(header) + '\n'
        f.write(header)
        f.write(sep)
        for idx, rec in enumerate(results, 1):
            strain, om1, om2, atoms, c1p, c1q, c2p, c2q, eps1, eps2 = rec
            i11, i12 = c1p; i21, i22 = c1q
            j11, j12 = c2p; j21, j22 = c2q
            f.write(f"|{idx:4d} | {strain:8.5f} | {atoms:5d} | {om1:3d}/{om2:<3d} | "
                    f"{i11:4d} {i12:4d} | {i21:4d} {i22:4d} | {j11:4d} {j12:4d} | "
                    f"{j21:4d} {j22:4d} | {eps1:8.2e} | {eps2:8.2e} |\n")

    # Console output
    print("Results:")
    print(header.strip())
    print(sep.strip())
    for idx, rec in enumerate(results, 1):
        strain, om1, om2, atoms, c1p, c1q, c2p, c2q, eps1, eps2 = rec
        i11, i12 = c1p; i21, i22 = c1q
        j11, j12 = c2p; j21, j22 = c2q
        print(f"|{idx:4d} | {strain:8.5f} | {atoms:5d} | {om1:3d}/{om2:<3d} | "
              f"{i11:4d} {i12:4d} | {i21:4d} {i22:4d} | {j11:4d} {j12:4d} | "
              f"{j21:4d} {j22:4d} | {eps1:8.2e} | {eps2:8.2e} |")

if __name__ == '__main__':
    main()