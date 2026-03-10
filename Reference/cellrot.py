import numpy as np
import sys
import os

def parse_poscar(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Extracting the lattice vectors (lines 3-5)
    lattice_vectors = np.array([list(map(float, lines[i].split())) for i in range(2, 5)])

    # Extracting atomic species and counts (lines 6 and 7)
    atom_types = lines[5].strip()  # Keep atom types exactly as they are
    atom_counts = lines[6].strip()  # Keep atom counts exactly as they are

    # Read coordinate type from line 8 (index 7)
    coord_type = lines[7].strip().lower()

    # Total number of atoms (sum of counts)
    num_atoms = sum(map(int, atom_counts.split()))
    # Read atomic positions (starting from line 9)
    atomic_positions = np.array([list(map(float, lines[i].split())) for i in range(8, 8 + num_atoms)])

    # Return positions as-is along with the coordinate type (Cartesian or Direct)
    return lattice_vectors, atomic_positions, atom_types, atom_counts, coord_type

def rotate_structure(atomic_positions, lattice_vectors, angle_deg):
    # Convert angle from degrees to radians
    angle_rad = np.radians(angle_deg)
    
    # Rotation matrix around the Z-axis
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    
    # Apply the rotation to the atomic positions
    rotated_positions = np.dot(atomic_positions, rotation_matrix.T)
    # Apply the rotation to the lattice vectors
    rotated_lattice = np.dot(lattice_vectors, rotation_matrix.T)
    
    return rotated_positions, rotated_lattice

def raise_structure(atomic_positions, height):
    # Raise the structure by the given height (along the z-axis)
    atomic_positions[:, 2] += height
    return atomic_positions

def write_poscar(file_path, lattice_vectors, atomic_positions, atom_types, atom_counts):
    with open(file_path, 'w') as f:
        # Write header and scale factor
        f.write('Generated POSCAR\n')
        f.write('1.0\n')
        
        # Write the (rotated) lattice vectors
        for vec in lattice_vectors:
            f.write('  ' + '  '.join(f'{x: .16f}' for x in vec) + '\n')

        # Write atom types and counts exactly as in the input file
        f.write(f"{atom_types}\n")
        f.write(f"{atom_counts}\n")
        
        # Write coordinate system as Direct
        f.write('Direct\n')
        for pos in atomic_positions:
            f.write('  ' + '  '.join(f'{x: .16f}' for x in pos) + '\n')

def main():
    if len(sys.argv) != 4:
        print("Usage: python cellrot.py <input_poscar> <angle> <height>")
        sys.exit(1)

    input_poscar = sys.argv[1]
    angle = float(sys.argv[2])
    height = float(sys.argv[3])

    # Step 1: Parse the POSCAR file without converting Cartesian to direct yet.
    lattice_vectors, atomic_positions, atom_types, atom_counts, coord_type = parse_poscar(input_poscar)

    # Step 2: Rotate the atomic positions and lattice vectors
    rotated_positions, rotated_lattice = rotate_structure(atomic_positions, lattice_vectors, angle)

    # Step 3: Raise the atomic positions by the specified height
    raised_positions = raise_structure(rotated_positions, height)

    # Step 4: If the input was in Cartesian, convert the rotated+raised positions to direct coordinates
    if coord_type.startswith('c'):  # 'cartesian'
        # Compute the inverse of the rotated lattice matrix
        inv_rotated = np.linalg.inv(rotated_lattice)
        # Convert from Cartesian to direct: direct = Cartesian * inv(rotated_lattice)
        final_positions = np.dot(raised_positions, inv_rotated)
    else:
        final_positions = raised_positions

    # Step 5: Write the modified structure to a new POSCAR file (always using Direct coordinates)
    output_poscar = f"{os.path.splitext(input_poscar)[0]}_{int(angle)}.vasp"
    write_poscar(output_poscar, rotated_lattice, final_positions, atom_types, atom_counts)

    print('                                                                                  \n'
      '===================================================================================== \n'
      '             cellrot: Cell rotating code for DFT or other calculations                \n'
      '===================================================================================== \n'
      'Copyright (C) 2025       Sarwin Chandran                                              \n'
      '                         sarwin@jncasr.ac.in                                          \n'
      '===================================================================================== \n'
      '                                                                                      \n'
      '                     Cell Matching Code    "cellmatch.py"                             \n'
      '            >>> Calculates factors for commensuration of two unit cells <<<           \n'
      '                                                                                      \n')

    indent = '    '
    ############################################################################################################

    print(f"Modified POSCAR written to {output_poscar}")

if __name__ == "__main__":
    main()

