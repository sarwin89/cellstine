"""Geometric utilities for 2D lattice operations.

All functions assume the first two lattice vectors define the in-plane
cell; the third vector is ignored except when computing strains (where
we impute a unit z-component).

These routines are numerically heavy and have been written using
NumPy for performance.
"""

import numpy as np
import math
from typing import Tuple, List


def rotate_vector(v: np.ndarray, theta_rad: float) -> np.ndarray:
    """Rotate a 2D vector by ``theta_rad`` radians about the origin."""
    c = math.cos(theta_rad)
    s = math.sin(theta_rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def unit_area(v1: np.ndarray, v2: np.ndarray) -> float:
    """Absolute 2D cross product; area of parallelogram spanned by v1,v2."""
    return abs(v1[0] * v2[1] - v1[1] * v2[0])


def calc_strain(a1, b1, c1, a2, b2, c2) -> float:
    """Calculate the average linear strain between two unit cells.

    Parameters are 3‑vectors giving two basis vectors and a dummy z of
    1.0 (we ignore out‑of‑plane distortions).  The algorithm is the same
    as the legacy ``calc_strain`` used in earlier versions of the
    project.
    """
    # force planar z=1.0 for compatibility
    a1x, a1y, a1z = a1; b1x, b1y, b1z = b1; c1z = 1.0
    a2x, a2y, a2z = a2; b2x, b2y, b2z = b2; c2z = 1.0
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
    S = 0.5 * (E + E.T + E @ E.T)
    ev = np.linalg.eigvals(S)
    return math.sqrt((ev * ev).sum()) / 3.0


def gen_span(lattice: np.ndarray, nmax: int) -> np.ndarray:
    """Generate all integer linear combinations a*e1 + b*e2 with
    -nmax <= a,b <= nmax (excluding the zero vector).

    Returns array of shape (M,2) where M=(2*nmax+1)**2-1.
    """
    idx = np.arange(-nmax, nmax + 1)
    a, b = np.meshgrid(idx, idx)
    coeffs = np.vstack((a.ravel(), b.ravel())).T
    coeffs = coeffs[~np.all(coeffs == 0, axis=1)]
    return coeffs @ lattice[:2, :2]


def filter_unique_directions(vectors: np.ndarray) -> np.ndarray:
    """Remove mirror duplicates; keep one representative for each direction."""
    dirs = np.array([np.sign(v) * np.abs(v) for v in vectors])
    # round to avoid floating point jitter before unique
    return np.unique(np.round(dirs, 6), axis=0)


def find_length_matches(span1: np.ndarray, span2: np.ndarray,
                        tol: float = 1e-5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return pairs of independent vectors from the two spans sharing length.

    Independence is checked by verifying the rank of the two vectors is 2.
    """
    n1 = np.linalg.norm(span1, axis=1)
    n2 = np.linalg.norm(span2, axis=1)
    matches = np.isclose(n1[:, None], n2[None, :], atol=tol)
    pairs = []
    for i in range(matches.shape[0]):
        for j in range(matches.shape[1]):
            if matches[i, j]:
                v1 = span1[i]; v2 = span2[j]
                if not np.allclose(v1, v2) and np.linalg.matrix_rank(np.vstack((v1, v2)).T) == 2:
                    pairs.append((v1, v2))
    return pairs


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle (degrees) between two vectors."""
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos = max(-1.0, min(1.0, cos))
    return math.degrees(math.acos(cos))


def gen_pairs(rot1: np.ndarray, lat2_2d: np.ndarray, nidx: int, tol: float):
    """Generate matching vector pairs between rotated cell1 and cell2.

    Copied from the prior cellmatch implementation; returns a list of
    dictionaries holding coefficient pairs, vectors and error.
    """
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


def build_supercells(pairs, lin_tol, rot1, lat2, cnt1, cnt2, angle=None):
    """Given matched pairs, produce a list of supercell candidates.

    The returned tuple contents match the legacy format; see
    :func:`moire.finder.find_supercells` for details.
    """
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
            # compute a few strain measures
            strain_avg = calc_strain([v1[0],v1[1],0], [v2[0],v2[1],0], [0,0,1],
                                     [g1[0],g1[1],0], [g2[0],g2[1],0], [0,0,1])
            n_v1 = np.linalg.norm(v1); n_v2 = np.linalg.norm(v2)
            n_g1 = np.linalg.norm(g1); n_g2 = np.linalg.norm(g2)
            s1 = math.sqrt(((n_g1/n_v1 - 1)**2 + (n_g2/n_v2 - 1)**2)/2) if n_v1>0 and n_v2>0 else 0.0
            s2 = math.sqrt(((n_v1/n_g1 - 1)**2 + (n_v2/n_g2 - 1)**2)/2) if n_g1>0 and n_g2>0 else 0.0
            atoms = round(cnt1 * om1 + cnt2 * om2)
            eps1 = sub[p]['eps']; eps2 = sub[q]['eps']
            results.append((strain_avg, s1, s2, om1, om2, atoms,
                            c1p, c1q, c2p, c2q, eps1, eps2, angle))
    return sorted(results, key=lambda x: x[0])
