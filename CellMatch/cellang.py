#!/usr/bin/env python
import numpy as np
import math
import argparse
import sys
import matplotlib.pyplot as plt

##########################
# Utility Functions
##########################
def rotate_vector_2d(vector, angle_rad):
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    return np.array([vector[0] * cos_theta - vector[1] * sin_theta,
                     vector[0] * sin_theta + vector[1] * cos_theta])

def unit_cell_area_2d(v1, v2):
    return abs(np.cross(v1, v2))

def is_integer_combination(target_vector, basis_vectors, tolerance=1e-9):
    matrix = np.array(basis_vectors).T  # 2x2 matrix with columns = basis vectors
    coeffs = np.linalg.inv(matrix) @ target_vector
    return np.allclose(coeffs, np.round(coeffs), atol=tolerance), np.round(coeffs).astype(int)

def find_commensurate_superlattice(lattice1_vectors, lattice2_vectors,
                                   angle_range=(20.01, 20.09),
                                   angle_step=0.001,
                                   search_limit=10,
                                   mismatch_tolerance=1e-3):
    a1, a2 = lattice1_vectors
    candidates = []  # collect all valid solutions

    for angle_deg in np.arange(angle_range[0], angle_range[1] + angle_step, angle_step):
        angle_rad = math.radians(angle_deg)
        # Rotate the second lattice vectors by the current angle
        rotated_lattice2_vectors = [rotate_vector_2d(v, angle_rad) for v in lattice2_vectors]

        # Generate potential superlattice vectors as integer combinations of lattice1
        possible_superlattice_vectors = [
            (m11 * a1 + m12 * a2, (m11, m12))
            for m11 in range(-search_limit, search_limit + 1)
            for m12 in range(-search_limit, search_limit + 1)
            if not (m11 == 0 and m12 == 0)
        ]

        for s1_candidate, m1_coeffs in possible_superlattice_vectors:
            is_commensurate_s1, n1_coeffs = is_integer_combination(s1_candidate, rotated_lattice2_vectors, tolerance=mismatch_tolerance)
            if is_commensurate_s1:
                for m21 in range(-search_limit, search_limit + 1):
                    for m22 in range(-search_limit, search_limit + 1):
                        if m21 == 0 and m22 == 0:
                            continue
                        v2_candidate = m21 * a1 + m22 * a2
                        if np.allclose(v2_candidate, s1_candidate):  # Avoid duplicate vector
                            continue
                        is_commensurate_s2, n2_coeffs = is_integer_combination(v2_candidate, rotated_lattice2_vectors, tolerance=mismatch_tolerance)
                        if is_commensurate_s2 and abs(np.cross(s1_candidate, v2_candidate)) > 1e-9:
                            area = unit_cell_area_2d(s1_candidate, v2_candidate)
                            M1 = np.array([m1_coeffs, (m21, m22)]).T
                            M2 = np.array([n1_coeffs, n2_coeffs]).T
                            candidate = {
                                "superlattice_vectors": [s1_candidate, v2_candidate],
                                "M1": M1,
                                "M2": M2,
                                "area": area,
                                "angle": angle_deg
                            }
                            candidates.append(candidate)
    if candidates:
        # Select the candidate with the smallest area
        best_candidate = min(candidates, key=lambda x: x["area"])
        return best_candidate
    else:
        return None

##########################
# POSCAR Reading Functions
##########################
def parse_poscar(file_path):
    """
    Reads a POSCAR file and returns:
      - lattice_vectors: full 3D lattice (as a NumPy array)
      - atomic_positions: atomic positions (as a NumPy array)
      - atom_types: line containing atom symbols
      - atom_counts: line containing atom counts
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Lines 3-5 hold the lattice vectors.
    lattice_vectors = np.array([list(map(float, lines[i].split())) for i in range(2, 5)])

    # Lines 6 and 7 contain atom types and counts.
    atom_types = lines[5].strip()
    atom_counts = lines[6].strip()
    
    num_atoms = sum(map(int, atom_counts.split()))
    # Atomic positions start at line 9
    atomic_positions = np.array([list(map(float, lines[i].split())) for i in range(8, 8 + num_atoms)])

    return lattice_vectors, atomic_positions, atom_types, atom_counts

def extract_in_plane_lattice(poscar_file):
    """
    Reads the POSCAR file and returns the in-plane lattice vectors (first two, taking only x,y)
    and the total number of atoms.
    """
    lattice_vectors, _, _, atom_counts = parse_poscar(poscar_file)
    in_plane = [np.array(lattice_vectors[0][:2]), np.array(lattice_vectors[1][:2])]
    return in_plane, sum(map(int, atom_counts.split()))

##########################
# Drawing Function
##########################
def draw_moire_pattern(original_lattice, rotated_lattice, superlattice_vectors, search_limit, angle_found):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,8))
    
    # Define superlattice cell corners from the two superlattice vectors
    v1 = superlattice_vectors[0]
    v2 = superlattice_vectors[1]
    cell_corners = np.array([[0, 0], v1, v1 + v2, v2])
    
    # Use an expanded range for lattice points based on search_limit.
    idx_range = range(-search_limit * 2, search_limit * 2 + 1)
    
    # Collect all lattice points from both original and rotated lattices.
    all_points = []
    for i in idx_range:
        for j in idx_range:
            pt_orig = i * original_lattice[0] + j * original_lattice[1]
            pt_rot = i * rotated_lattice[0] + j * rotated_lattice[1]
            all_points.append(pt_orig)
            all_points.append(pt_rot)
    all_points = np.array(all_points)
    
    # Include the superlattice cell corners in the bounding box calculation
    all_points = np.vstack((all_points, cell_corners))
    
    # Determine overall bounding box from all points
    x_min = np.min(all_points[:, 0])
    x_max = np.max(all_points[:, 0])
    y_min = np.min(all_points[:, 1])
    y_max = np.max(all_points[:, 1])
    
    # Increase the margin relative to the range
    margin_factor = 0.2
    margin_x = margin_factor * (x_max - x_min) if (x_max - x_min) != 0 else 1.0
    margin_y = margin_factor * (y_max - y_min) if (y_max - y_min) != 0 else 1.0
    x_lim = (x_min - margin_x, x_max + margin_x)
    y_lim = (y_min - margin_y, y_max + margin_y)
    
    # Plot original lattice points (blue)
    for i in idx_range:
        for j in idx_range:
            point = i * original_lattice[0] + j * original_lattice[1]
            ax.plot(point[0], point[1], 'bo', markersize=2, alpha=0.5)
    
    # Plot rotated lattice points (red)
    for i in idx_range:
        for j in idx_range:
            point = i * rotated_lattice[0] + j * rotated_lattice[1]
            ax.plot(point[0], point[1], 'ro', markersize=2, alpha=0.5)
    
    # Draw the superlattice unit cell (black)
    cell_path = np.vstack((cell_corners, cell_corners[0]))
    ax.plot(cell_path[:, 0], cell_path[:, 1], 'k-', linewidth=2)
    
    # Annotate the plot with the found rotation angle
    ax.text(0.05, 0.95, f"Rotation Angle: {angle_found:.3f}°",
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8))
    
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Moiré Pattern and Superlattice")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()

##########################
# Main Function
##########################
def main():
    parser = argparse.ArgumentParser(description="Find a commensurate superlattice from a POSCAR file (for 2D materials).")
    parser.add_argument("poscar", help="Input POSCAR file for the material")
    parser.add_argument("angle_lower", type=float, help="Lower bound for rotation angle range (in degrees)")
    parser.add_argument("angle_upper", type=float, help="Upper bound for rotation angle range (in degrees)")
    parser.add_argument("--angle_step", type=float, default=0.001, help="Angle step for the search (in degrees)")
    parser.add_argument("--search_limit", type=int, default=15, help="Limit for integer coefficient search")
    parser.add_argument("--mismatch_tolerance", type=float, default=1e-3, help="Tolerance for integer combination mismatch")
    parser.add_argument("--draw", type=int, default=0, help="If 1, plot the moiré pattern and superlattice")
    args = parser.parse_args()

    # Extract in-plane lattice vectors and original number of atoms from the POSCAR
    in_plane_lattice, original_atom_count = extract_in_plane_lattice(args.poscar)
    area_original = unit_cell_area_2d(in_plane_lattice[0], in_plane_lattice[1])

    angle_range = (args.angle_lower, args.angle_upper)

    # Search for a commensurate superlattice
    result = find_commensurate_superlattice(in_plane_lattice, in_plane_lattice,
                                            angle_range=angle_range,
                                            angle_step=args.angle_step,
                                            search_limit=args.search_limit,
                                            mismatch_tolerance=args.mismatch_tolerance)
    if result is None:
        print("No commensurate superlattice found within the given parameters.")
        sys.exit(1)

    # Compute number of atoms in the supercell from the area ratio
    area_supercell = result["area"]
    num_atoms_supercell = int(round((area_supercell / area_original) * original_atom_count))

    # Print results
    print("=== Commensurate Superlattice Found ===")
    print("Superlattice Vectors (in-plane):")
    for vec in result["superlattice_vectors"]:
        print(vec)
    print("\nM1 Coefficients (from original lattice):")
    print(result["M1"])
    print("\nM2 Coefficients (from rotated lattice):")
    print(result["M2"])
    print("\nSupercell Area: {:.6f}".format(area_supercell))
    print("Rotation Angle: {:.3f} degrees".format(result["angle"]))
    print("Number of atoms in the supercell: {}".format(num_atoms_supercell))

    # If draw flag is set, plot the moiré pattern and superlattice
    if args.draw == 1:
        found_angle = result["angle"]
        # Rotate the original in-plane lattice by the found angle to obtain the rotated lattice
        rotated_lattice = [rotate_vector_2d(v, math.radians(found_angle)) for v in in_plane_lattice]
        draw_moire_pattern(in_plane_lattice, rotated_lattice, result["superlattice_vectors"], args.search_limit, result["angle"])

if __name__ == "__main__":
    main()

