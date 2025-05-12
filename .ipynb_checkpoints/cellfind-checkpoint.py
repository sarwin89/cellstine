#!/usr/bin/env python3
import numpy as np
from scipy.linalg import null_space
import argparse

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

def find_common_vectors(lattice_vectors_1, lattice_vectors_2, tolerance=1e-5):
    # Stack the lattice vectors as columns for the matrix M
    M = np.column_stack((lattice_vectors_1[:, 0], lattice_vectors_1[:, 1], -lattice_vectors_2[:, 0], -lattice_vectors_2[:, 1]))
    
    # Find the null space (solution to M @ [a, b, c, d] = 0)
    ns = null_space(M)
    
    # Print the common vectors from the null space
    if ns.size == 0:
        print("The only common vector is the zero vector.")
    else:
        print("The intersection of the spans is spanned by:")
        for i in range(ns.shape[1]):
            a, b, c, d = ns[:, i]
            # Form the vector in the intersection
            common_vec = a * lattice_vectors_1[:, 0] + b * lattice_vectors_1[:, 1]
            
            # Normalize for nicer output (optional)
            norm = np.linalg.norm(common_vec)
            if norm > tolerance:
                common_vec = common_vec / norm
                print(f"Basis vector {i+1}: {common_vec}")
            else:
                print(f"Basis vector {i+1} is effectively the zero vector.")
        
        print("\nAny linear combination of these basis vectors is in the intersection.")

def main():
    parser = argparse.ArgumentParser(description="Find common vectors in the spans of two POSCAR files.")
    parser.add_argument('poscar1', type=str, help="Path to the first POSCAR file.")
    parser.add_argument('poscar2', type=str, help="Path to the second POSCAR file.")
    parser.add_argument('index_limit', type=int, help="Limit on the number of lattice vectors to analyze.")
    parser.add_argument('tolerance', type=float, default=1e-5, help="Tolerance for comparing vectors.")
    
    args = parser.parse_args()
    
    # Parse the POSCAR files
    lattice_vectors_1, _, _, _ = parse_poscar(args.poscar1)
    lattice_vectors_2, _, _, _ = parse_poscar(args.poscar2)
    
    # Consider the index limit for the lattice vectors
    lattice_vectors_1 = lattice_vectors_1[:args.index_limit]
    lattice_vectors_2 = lattice_vectors_2[:args.index_limit]
    
    # Find common vectors
    find_common_vectors(lattice_vectors_1, lattice_vectors_2, args.tolerance)

if __name__ == "__main__":
    main()
