#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
import argparse
import math, sys

def rotate_vector_2d(vector, angle_rad):
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    return np.dot(rotation_matrix, vector)

def unit_cell_area_2d(v1, v2):
    return abs(np.cross(v1, v2))

def parse_poscar(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    scaling_factor = float(lines[1].strip())
    lattice_vectors = np.array([list(map(float, lines[i].split())) for i in range(2, 5)]) * scaling_factor
    
    atom_types = lines[5].strip().split()
    atom_counts = list(map(int, lines[6].strip().split()))
    
    num_atoms = sum(atom_counts)
    atomic_positions = np.array([list(map(float, lines[i].split()[:3])) for i in range(8, 8 + num_atoms)])
    
    return lattice_vectors, atomic_positions, atom_types, atom_counts

def find_commensurate_superlattice_single(angle_deg, lattice1_vectors, lattice2_vectors, search_limit, mismatch_tolerance):
    angle_rad = np.deg2rad(angle_deg)
    rotated_lattice1 = np.array([rotate_vector_2d(vec, angle_rad) for vec in lattice1_vectors])
    
    candidate_pairs = []
    # Loop over integer coefficient pairs for the rotated cell1 and for cell2.
    # For rotated cell1, use coefficients (m, n), and for cell2, (p, q).
    for m in range(-search_limit, search_limit + 1):
        for n in range(-search_limit, search_limit + 1):
            if m == 0 and n == 0:
                continue
            # Build candidate vector V for rotated cell1
            V = m * rotated_lattice1[0] + n * rotated_lattice1[1]
            normV = np.linalg.norm(V)
            if normV < 1e-8:
                continue
            for p in range(-search_limit, search_limit + 1):
                for q in range(-search_limit, search_limit + 1):
                    if p == 0 and q == 0:
                        continue
                    # Build candidate vector W for cell2 (no rotation)
                    W = p * lattice2_vectors[0] + q * lattice2_vectors[1]
                    normW = np.linalg.norm(W)
                    if normW < 1e-8:
                        continue
                    # Compute the relative mismatch between V and W.
                    error = np.linalg.norm(V - W) / (normV + normW)
                    if error < mismatch_tolerance:
                        candidate_pairs.append({
                            "coeff_cell1": np.array([m, n]),
                            "coeff_cell2": np.array([p, q]),
                            "vector1": V,
                            "vector2": W,
                            "error": error,
                            "length": (normV + normW) / 2.0
                        })
    
    superlattice_candidates = []
    N = len(candidate_pairs)
    tol_det = 1e-8  # Tolerance for checking linear independence
    # Combine two candidate pairs to form a full 2D supercell.
    for i in range(N):
        for j in range(i + 1, N):
            V1 = candidate_pairs[i]["vector1"]
            V2 = candidate_pairs[j]["vector1"]
            # Check that the candidate vectors from rotated cell1 are linearly independent.
            if abs(np.linalg.det(np.array([V1, V2]))) < tol_det:
                continue
            W1 = candidate_pairs[i]["vector2"]
            W2 = candidate_pairs[j]["vector2"]
            # Check that the corresponding cell2 vectors are linearly independent.
            if abs(np.linalg.det(np.array([W1, W2]))) < tol_det:
                continue
            
            # Compute the average candidate strain from the two candidate errors.
            strain = (candidate_pairs[i]["error"] + candidate_pairs[j]["error"]) / 2.0
            
            # Calculate the area of the candidate supercell for each cell and then average them.
            area1 = abs(np.linalg.det(np.array([V1, V2])))
            area2 = abs(np.linalg.det(np.array([W1, W2])))
            area = (area1 + area2) / 2.0
            
            # Average the candidate vectors to represent the final superlattice vectors.
            super_vector1 = (V1 + W1) / 2.0
            super_vector2 = (V2 + W2) / 2.0
            
            # Combine the coefficients for each candidate pair into a 4-element array.
            coeff_M1 = np.concatenate((candidate_pairs[i]["coeff_cell1"], candidate_pairs[i]["coeff_cell2"]))
            coeff_M2 = np.concatenate((candidate_pairs[j]["coeff_cell1"], candidate_pairs[j]["coeff_cell2"]))
            
            superlattice_candidates.append({
                "angle": angle_deg,
                "strain": strain,
                "area": area,
                "superlattice_vectors": np.array([super_vector1, super_vector2]),
                "coefficients": {
                    "M1": coeff_M1,
                    "M2": coeff_M2
                }
            })
    
    # Uniqueness filtering: remove nearly identical candidates.
    # Tolerances below are similar in spirit to those used in cellmatch.py.
    tolerance_strain = 1e-4
    tolerance_area_ratio = 1e-5  # relative difference in area considered negligible
    # First, sort candidates by strain (lowest first)
    superlattice_candidates.sort(key=lambda x: x["strain"])
    
    unique_candidates = []
    for candidate in superlattice_candidates:
        is_duplicate = False
        for uniq in unique_candidates:
            if abs(candidate["strain"] - uniq["strain"]) < tolerance_strain:
                # Compare the area ratio between candidates; if nearly 1, consider them duplicates.
                if uniq["area"] != 0:
                    area_ratio = candidate["area"] / uniq["area"]
                else:
                    area_ratio = 0
                if abs(area_ratio - 1.0) < tolerance_area_ratio:
                    is_duplicate = True
                    break
        if not is_duplicate:
            unique_candidates.append(candidate)
    
    # Finally, sort the unique candidates by strain (lowest first).
    unique_candidates.sort(key=lambda x: x["strain"])
    return unique_candidates

def find_commensurate_superlattice(lattice1_vectors, lattice2_vectors, angle_range, angle_step, search_limit, mismatch_tolerance, num_processes):
    angles = np.arange(angle_range[0], angle_range[1] + angle_step, angle_step)
    with multiprocessing.Pool(processes=num_processes) as pool:
        func = partial(find_commensurate_superlattice_single, lattice1_vectors=lattice1_vectors, 
                       lattice2_vectors=lattice2_vectors, search_limit=search_limit, 
                       mismatch_tolerance=mismatch_tolerance)
        results = pool.map(func, angles)
    
    supercells = [item for sublist in results for item in sublist]
    return sorted(supercells, key=lambda x: x["area"])

def main():
    parser = argparse.ArgumentParser(description="Find commensurate supercells between two POSCAR files.")
    parser.add_argument("poscar1", help="First POSCAR file")
    parser.add_argument("poscar2", help="Second POSCAR file")
    parser.add_argument("angle_lower", type=float, help="Lower bound of angle range (degrees)")
    parser.add_argument("angle_upper", type=float, help="Upper bound of angle range (degrees)")
    parser.add_argument("--angle_step", type=float, default=0.001, help="Angle step (degrees)")
    parser.add_argument("--nindex", type=int, default=10, help="Range of integer coefficients")
    parser.add_argument("--tolerance", type=float, default=1e-5, help="Matching tolerance")
    parser.add_argument("--draw", type=int, default=0, help="Draw Moiré pattern (1=yes)")
    parser.add_argument("--output", type=str, default="results.dat", help="Output file")
    parser.add_argument("--input_dat", type=str, help="Optional input .dat file with precomputed results")
    parser.add_argument("--processes", type=int, default=6, help="Number of processes")
    args = parser.parse_args()

    lattice1, coords1, types1, counts1 = parse_poscar(args.poscar1)
    lattice2, coords2, types2, counts2 = parse_poscar(args.poscar2)

    lattice1_vectors = lattice1[:2, :2]
    lattice2_vectors = lattice2[:2, :2]

    supercells = find_commensurate_superlattice(
        lattice1_vectors, lattice2_vectors,
        angle_range=(args.angle_lower, args.angle_upper),
        angle_step=args.angle_step,
        search_limit=args.nindex,
        mismatch_tolerance=args.tolerance,
        num_processes=args.processes
    )

    with open(args.output, 'w') as f:
        f.write(f"Unit Cell 1: {args.poscar1}\n")
        f.write(f"Unit Cell 2: {args.poscar2}\n\n")
        for i, supercell in enumerate(supercells, 1):
            f.write(f"Candidate {i}:\n")
            f.write(f"Angle: {supercell['angle']:.6f} degrees\n")
            f.write(f"Strain: {supercell['strain']:.4f}\n")
            f.write(f"Area: {supercell['area']:.4f}\n")
            f.write(f"Vectors: {supercell['superlattice_vectors']}\n")
            f.write(f"Coefficients: M1={supercell['coefficients']['M1'].tolist()}, M2={supercell['coefficients']['M2'].tolist()}\n\n")

    for i, supercell in enumerate(supercells, 1):
        print(f"  Candidate {i}:")
        print(f"  Angle: {supercell['angle']:.6f} degrees")
        print(f"  Strain: {supercell['strain']:.4f}")
        print(f"  Area: {supercell['area']:.4f}")
        print(f"  Vectors: {supercell['superlattice_vectors']}")
        print(f"  Coefficients: M1={supercell['coefficients']['M1'].tolist()}, M2={supercell['coefficients']['M2'].tolist()}\n")

if __name__ == '__main__':
    main()