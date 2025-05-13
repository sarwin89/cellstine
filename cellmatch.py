#!/usr/bin/env python3
import numpy as np
import argparse
import math
import multiprocessing as mp

def rotate_vector_2d(v, theta):
    """Rotate 2D vector v by angle theta (radians)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])

def unit_area(v1, v2):
    """Calculate the absolute 2D cross product (area spanned by v1 and v2)."""
    return abs(v1[0] * v2[1] - v1[1] * v2[0])

def parse_poscar(path):
    """Parse a VASP POSCAR/CONTCAR file: return lattice (3x3), coordinates (Nx3), and counts list."""
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    scale = float(lines[1].strip())
    lat = np.array([list(map(float, lines[i].split())) for i in (2, 3, 4)]) * scale
    counts = list(map(int, lines[6].split()))
    nats = sum(counts)
    coords = np.array([list(map(float, lines[i].split()[:3]))
                       for i in range(8, 8 + nats)])
    return lat, coords, counts

# --- Candidate Pair Generation ---
def gen_pairs(rot1, lat2_2d, nidx, tol):
    rot1 = np.asarray(rot1)[:, :2]
    lat2_2d = np.asarray(lat2_2d)[:, :2]
    idx = np.arange(-nidx, nidx + 1)
    i1, i2 = np.meshgrid(idx, idx)
    c1 = np.vstack((i1.ravel(), i2.ravel())).T
    c1 = c1[~np.all(c1 == 0, axis=1)]
    j1, j2 = np.meshgrid(idx, idx)
    c2 = np.vstack((j1.ravel(), j2.ravel())).T
    c2 = c2[~np.all(c2 == 0, axis=1)]
    V = c1[:, 0, None] * rot1[0] + c1[:, 1, None] * rot1[1]
    G = c2[:, 0, None] * lat2_2d[0] + c2[:, 1, None] * lat2_2d[1]
    nV = np.linalg.norm(V, axis=1)
    nG = np.linalg.norm(G, axis=1)
    D = V[:, None, :] - G[None, :, :]
    err = np.linalg.norm(D, axis=2) / (nV[:, None] + nG[None, :])
    ai, bi = np.nonzero(err < tol)
    pairs = []
    for ia, ib in zip(ai, bi):
        pairs.append({
            'c1': c1[ia].tolist(),
            'c2': c2[ib].tolist(),
            'v1': V[ia].tolist(),
            'v2': G[ib].tolist(),
            'err': float(err[ia, ib]),
            'length': 0.5 * (nV[ia] + nG[ib])
        })
    pairs.sort(key=lambda p: p['err'])
    return pairs

# --- Strain Calculation ---
def calc_strain(a1, b1, c1, a2, b2, c2):
    # enforce z = 1
    a1x, a1y, a1z = a1; b1x, b1y, b1z = b1; c1z = 1.0
    a2x, a2y, a2z = a2; b2x, b2y, b2z = b2; c2z = 1.0
    M1 = np.array([
        [a1x*a1x + a1y*a1y + a1z*a1z, a1x*b1x + a1y*b1y + a1z*b1z, a1x*c1[0] + a1y*c1[1] + a1z*c1z],
        [a1x*b1x + a1y*b1y + a1z*b1z, b1x*b1x + b1y*b1y + b1z*b1z, b1x*c1[0] + b1y*c1[1] + b1z*c1z],
        [a1x*c1[0] + a1y*c1[1] + a1z*c1z, c1[0]*b1x + c1[1]*b1y + c1z*b1z, c1[0]*c1[0] + c1[1]*c1[1] + c1z*c1z]
    ])
    M2 = np.array([
        [a2x*a2x + a2y*a2y + a2z*a2z, a2x*b2x + a2y*b2y + a2z*b2z, a2x*c2[0] + a2y*c2[1] + a2z*c2z],
        [a2x*b2x + a2y*b2y + a2z*b2z, b2x*b2x + b2y*b2y + b2z*b2z, b2x*c2[0] + b2y*c2[1] + b2z*c2z],
        [a2x*c2[0] + a2y*c2[1] + a2z*c2z, c2[0]*b2x + c2[1]*b2y + c2z*b2z, c2[0]*c2[0] + c2[1]*c2[1] + c2z*c2z]
    ])
    rt1 = np.linalg.cholesky(M1).T
    rt2 = np.linalg.cholesky(M2).T
    E = rt2 @ np.linalg.inv(rt1) - np.eye(3)
    S = 0.5 * (E + E.T + E @ E.T)
    ev = np.linalg.eigvals(S)
    return math.sqrt((ev * ev).sum()) / 3.0

# --- Build Supercells ---
def build_supercells(pairs, lin_tol, rot1, lat2_2d, cnt1, cnt2, angle):
    """From matched vector pairs, build supercell candidates at a given rotation angle."""
    errs = np.array([p['err'] for p in pairs])
    k = int((errs <= lin_tol).sum())
    if k == 0:
        return []
    sub = pairs[:k]

    V = np.array([p['v1'] for p in sub])
    G = np.array([p['v2'] for p in sub])
    E = errs[:k]

    i, j = np.triu_indices(k, 1)
    Vi, Vj = V[i], V[j]
    Gi, Gj = G[i], G[j]

    d1 = Vi[:,0]*Vj[:,1] - Vi[:,1]*Vj[:,0]
    d2 = Gi[:,0]*Gj[:,1] - Gi[:,1]*Gj[:,0]
    A1 = unit_area(rot1[0], rot1[1])
    A2 = unit_area(lat2_2d[0], lat2_2d[1])

    o1 = np.round(np.abs(d1 / A1))
    o2 = np.round(np.abs(d2 / A2))
    valid = (o1 > 0.1) & (o2 > 0.1)
    idx = np.nonzero(valid)[0]

    res = []
    for u in idx:
        p, q = i[u], j[u]
        v1, v2 = V[p], V[q]
        g1, g2 = G[p], G[q]
        strain = calc_strain([v1[0],v1[1],0], [v2[0],v2[1],0], [0,0,1],
                             [g1[0],g1[1],0], [g2[0],g2[1],0], [0,0,1])
        length = np.linalg.norm(v1) * np.linalg.norm(v2)
        atoms = round(cnt1 * o1[u] + cnt2 * o2[u])
        # extract cell1 expansion indices for both basis vectors
        c1_p = tuple(pairs[p]['c1'])
        c1_q = tuple(pairs[q]['c1'])
        data = [int(o1[u]), int(o2[u]), length, atoms,
                [c1_p, c1_q], float(E[p]), float(E[q]), angle]
        res.append([strain, data])
    return sorted(res, key=lambda x: x[0])

# --- Angle Scan and Search ---
def process_angle(angle, lat1, lat2, nidx, tol, lin_tol, cnt1, cnt2):
    theta = math.radians(angle)
    rot1  = np.array([rotate_vector_2d(v, theta) for v in lat1[:2,:2]])
    pairs = gen_pairs(rot1, lat2[:2,:2], nidx, tol)

    # —— Debug snippet starts here —— 
    # Build the full V and G arrays so we can inspect all errors
    idx  = np.arange(-nidx, nidx+1)
    c1   = np.vstack(np.meshgrid(idx, idx)).T.reshape(-1,2)
    c1   = c1[~np.all(c1==0, axis=1)]
    c2   = np.vstack(np.meshgrid(idx, idx)).T.reshape(-1,2)
    c2   = c2[~np.all(c2==0, axis=1)]

    V = c1 @ rot1
    G = c2 @ lat2[:2,:2]
    nV  = np.linalg.norm(V, axis=1)
    nG  = np.linalg.norm(G, axis=1)

    D = V[:,None,:] - G[None,:,:]
    errs = np.linalg.norm(D, axis=2) / (nV[:,None] + nG[None,:])

    # Mask out the trivial zero‐vector self‐matches
    nontrivial = errs[~np.eye(errs.shape[0], dtype=bool)]
    print(f"Angle {angle:.4f}° → min non-trivial rel‐error = {nontrivial.min():.6f}")
    # —— Debug snippet ends here ——

    return build_supercells(pairs, lin_tol, rot1, lat2[:2,:2], cnt1, cnt2, angle)


# --- CLI ---
def main():
    p = argparse.ArgumentParser(description="Find a common unit cell between two POSCAR files.")
    p.add_argument('pos1')
    p.add_argument('pos2')
    p.add_argument('angle_lower', type=float)
    p.add_argument('angle_upper', type=float)
    p.add_argument('--angle_step', type=float, default=0.001)
    p.add_argument('--nindex', type=int, default=10)
    p.add_argument('--tolerance', type=float, default=1e-5)
    p.add_argument('--lin_tol', type=float, default=1e-4)
    p.add_argument('--processes', type=int, default=6)
    p.add_argument('--output', default='results.dat')
    args = p.parse_args()

    lat1, _, cnt1 = parse_poscar(args.pos1)
    lat2, _, cnt2 = parse_poscar(args.pos2)
    tot1, tot2 = sum(cnt1), sum(cnt2)

    print(f"Scanning angles from {args.angle_lower}° to {args.angle_upper}° in steps of {args.angle_step}°...")
    angles = np.arange(args.angle_lower, args.angle_upper + args.angle_step, args.angle_step)
    with mp.Pool(args.processes) as pool:
        # prepare arguments for each angle to avoid lambda (pickleable)
        tasks = [(ang, lat1, lat2, args.nindex, args.tolerance, args.lin_tol, tot1, tot2) for ang in angles]
        all_results = pool.starmap(process_angle, tasks)
    flat = [item for sub in all_results for item in sub]

    # uniqueness filtering by strain and surf_ratio
    # direct flatten since data[-1] is angle, ignore it for filtering
    uniq = {}
    for strain, data in flat:
        key = (round(strain/args.lin_tol), data[0]/data[1])
        if key not in uniq or data[2] < uniq[key][1][2]:
            uniq[key] = (strain, data)
    results = list(uniq.values())
    # sort by strain
    results.sort(key=lambda x: x[0])

    # Write results
    with open(args.output, 'w') as f:
        f.write("| idx | angle (deg) | strain   | atoms | surf_ratio |  idx1  |  idx2  |\n")
        f.write("--------------------------------------------------------------\n")
        for i, (strain, data) in enumerate(results, 1):
            o1, o2, length, atoms, idxs, err1, err2, ang = data
            (i1, i2), (j1, j2) = idxs
            f.write(f"|{i:4d} | {ang:10.4f} | {strain:8.6f} | {atoms:5d} | {int(o1):3d}/{int(o2):<3d} | {i1:3d},{i2:3d} | {j1:3d},{j2:3d} |\n")

    # Console output
    print("Results:")
    print("| idx | angle (deg) | strain   | atoms | surf_ratio |  idx1  |  idx2  |")
    print("--------------------------------------------------------------")
    for i, (strain, data) in enumerate(results, 1):
        o1, o2, length, atoms, idxs, err1, err2, ang = data
        (i1, i2), (j1, j2) = idxs
        print(f"|{i:4d} | {ang:10.4f} | {strain:8.6f} | {atoms:5d} | {int(o1):3d}/{int(o2):<3d} | {i1:3d},{i2:3d} | {j1:3d},{j2:3d} |")

if __name__ == '__main__':
    main()