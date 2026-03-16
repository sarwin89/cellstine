import numpy as np
import argparse
import time
import math

start_time = time.time()

def parse_poscar(file_path):
    # ParSCARs the files to extract lattice vectors
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    scaling_factor = float(lines[1].strip())
    lattice_vectors = np.array([list(map(float, lines[i].split())) for i in range(2, 5)]) * scaling_factor
    return lattice_vectors

def generate_span(lattice_vectors, n_index):
    # Meshgrid OP, Efficient span generation for a and b coef
    a_values, b_values = np.meshgrid(np.linspace(-n_index, n_index, 2 * n_index + 1),
                                     np.linspace(-n_index, n_index, 2 * n_index + 1))
    a_values = a_values.flatten()
    b_values = b_values.flatten()
    
    span_vectors = a_values[:, None] * lattice_vectors[0] + b_values[:, None] * lattice_vectors[1]
    return span_vectors

def are_linearly_independent(v1, v2):
    matrix = np.vstack([v1, v2]).T
    return np.linalg.matrix_rank(matrix) == 2

def find_matching_length_vectors(span1, span2, tolerance=1e-5):
    lengths1 = np.linalg.norm(span1, axis=1)
    lengths2 = np.linalg.norm(span2, axis=1)
    
    matching_lengths = np.isclose(lengths1[:, None], lengths2[None, :], atol=tolerance)
    
    # Determines vectors with identical lengths present in both spans
    matching_vectors = []
    for i in range(matching_lengths.shape[0]):
        for j in range(matching_lengths.shape[1]):
            if matching_lengths[i, j] and not np.allclose(span1[i], span2[j]) and are_linearly_independent(span1[i], span2[j]):
                matching_vectors.append((span1[i], span2[j]))
    
    return matching_vectors

def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def get_unique_directions(span):
    # Remove mirror copies
    directions = np.array([np.sign(vec) * np.abs(vec) for vec in span])
    return np.unique(np.round(directions, decimals=6), axis=0)

def lcm(a, b):
    return abs(a * b) // math.gcd(a, b) 

def atom_count():

    return 

def main():
    # Argument parsing, non-verbal :P
    parser = argparse.ArgumentParser(description="Find vectors with the same length but not the same vector in the spans of two POSCAR files.")
    parser.add_argument('poscar1', type=str, help="Path to the first POSCAR file.")
    parser.add_argument('poscar2', type=str, help="Path to the second POSCAR file.")
    parser.add_argument('nindex', type=int, help="Maximum index for the coefficients a and b for the span.")
    parser.add_argument('--tolerance', type=float, default=1e-5, help="Tolerance for length comparison.")
    
    args = parser.parse_args()
    
    # Get lattices
    lattice_vectors_1 = parse_poscar(args.poscar1)
    lattice_vectors_2 = parse_poscar(args.poscar2)
    
    # Get spans
    span1 = generate_span(lattice_vectors_1, args.nindex)
    span2 = generate_span(lattice_vectors_2, args.nindex)

    # Get non-mirror spans
    unique_span1 = get_unique_directions(span1)
    unique_span2 = get_unique_directions(span2)

    # Angles of Unit Cells
    angle1 = calculate_angle(lattice_vectors_1[0], lattice_vectors_1[1])
    angle2 = calculate_angle(lattice_vectors_2[0], lattice_vectors_2[1])
    angle_lcm = lcm(int(round(angle1)), int(round(angle2)))
    upper_limit = angle_lcm / 2.0
    
    # Vectors swiping right
    matching_vectors = find_matching_length_vectors(unique_span1, unique_span2, args.tolerance)
    
    # Get angles
    angles = []
    for v1, v2 in matching_vectors:
        angle = calculate_angle(v1, v2)
        if 0 <= angle <= upper_limit:
            angles.append(round(angle, 4))
    
    angles_np = np.array(angles)
    unique_angles = np.unique(angles_np)
    sorted_angles = np.sort(unique_angles)

    if sorted_angles.size > 0:
        print(f"Supercell possible angles (in degrees) between (0° to {upper_limit}°):")
        print(sorted_angles)
    else:
        print(f"No supercell candidates found within the 0° to {upper_limit}° range.")

if __name__ == "__main__":
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
