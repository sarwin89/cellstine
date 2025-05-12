import numpy as np
import argparse
import time
start_time = time.time()

def parse_poscar(file_path):
    # Parses a POSCAR file to extract lattice vectors
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    scaling_factor = float(lines[1].strip())
    lattice_vectors = np.array([list(map(float, lines[i].split())) for i in range(2, 5)]) * scaling_factor
    
    atom_types = lines[5].strip().split()
    atom_counts = list(map(int, lines[6].strip().split()))
    
    num_atoms = sum(atom_counts)
    atomic_positions = np.array([list(map(float, lines[i].split()[:3])) for i in range(8, 8 + num_atoms)])
    
    return lattice_vectors, atomic_positions, atom_types, atom_counts

def generate_span(lattice_vectors, n_index):
    # Efficient span generation using meshgrid to create coefficients a and b
    a_values, b_values = np.meshgrid(np.linspace(-n_index, n_index, 2 * n_index + 1),
                                     np.linspace(-n_index, n_index, 2 * n_index + 1))
    a_values = a_values.flatten()
    b_values = b_values.flatten()
    
    # Generate the linear combinations
    span_vectors = np.array([a * lattice_vectors[0] + b * lattice_vectors[1] for a, b in zip(a_values, b_values)])
    
    return span_vectors

def find_matching_length_vectors(span1, span2, tolerance=1e-5):
    # Calculate lengths of vectors in both spans
    lengths1 = np.linalg.norm(span1, axis=1)
    lengths2 = np.linalg.norm(span2, axis=1)
    
    # Vectorized matching of lengths
    matching_lengths = np.isclose(lengths1[:, None], lengths2[None, :], atol=tolerance)
    
    # Extract pairs where lengths match but the vectors are not identical
    matching_vectors = []
    for i in range(matching_lengths.shape[0]):
        for j in range(matching_lengths.shape[1]):
            if matching_lengths[i, j] and not np.allclose(span1[i], span2[j]):
                matching_vectors.append((span1[i], span2[j]))
    
    return matching_vectors

def calculate_angle(v1, v2):
    # Vectorized dot product and angle calculation between two vectors
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)  # Ensure the cosine is within valid range
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def get_unique_directions(span):
    # Remove duplicates caused by opposite signs (normalize direction)
    directions = np.array([np.sign(vec) * np.abs(vec) for vec in span])
    unique_directions = np.unique(np.round(directions, decimals=6), axis=0)  # Round to remove precision errors
    return unique_directions

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Find vectors with the same length but not the same vector in the spans of two POSCAR files.")
    parser.add_argument('poscar1', type=str, help="Path to the first POSCAR file.")
    parser.add_argument('poscar2', type=str, help="Path to the second POSCAR file.")
    parser.add_argument('n_index', type=int, help="Maximum index for the coefficients a and b for the span.")
    parser.add_argument('tolerance', type=float, default=1e-5, help="Tolerance for length comparison.")
    
    args = parser.parse_args()
    
    # Parse POSCAR files to get lattice vectors
    lattice_vectors_1, _, _, _ = parse_poscar(args.poscar1)
    lattice_vectors_2, _, _, _ = parse_poscar(args.poscar2)
    
    # Generate the span for each set of vectors
    span1 = generate_span(lattice_vectors_1, args.n_index)
    span2 = generate_span(lattice_vectors_2, args.n_index)
    
    # Get unique directions (ignoring signs)
    unique_span1 = get_unique_directions(span1)
    unique_span2 = get_unique_directions(span2)
    
    # Find matching vectors with the same length but not the same vector
    matching_vectors = find_matching_length_vectors(unique_span1, unique_span2, args.tolerance)
    
    # Calculate the angles between the matching vectors and filter those between 0 and 60 degrees
    angles = []
    for v1, v2 in matching_vectors:
        angle = calculate_angle(v1, v2)
        if 0 <= angle <= 30:  # Keep only angles between 0 and 60 degrees
            angles.append(round(angle, 4))  # Round the angle to 4 decimal places
    
    # Convert the list to a numpy array and remove duplicates
    angles_np = np.array(angles)
    unique_angles = np.unique(angles_np)
    
    # Sort the angles in increasing order
    sorted_angles = np.sort(unique_angles)
    
    # Output the unique angles as a numpy array
    if sorted_angles.size > 0:
        print("Unique angles (in degrees) between vectors with matching lengths but not the same vector (0° to 60°):")
        print(sorted_angles)
    else:
        print("No matching vectors found within the 0° to 30° range.")

if __name__ == "__main__":
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
