#!/usr/bin/env python
# Copyright (C) 2014 Predrag Lazic
# This file is part of the CellMatch - code for finding a common unit cell
# of the two given cells
#
# CellMatch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CellMatch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CellMatch.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import numpy as np
from math import *
import os, sys, time
import pylab
from pylab import *
import matplotlib.pyplot as plt


def draw_unit_cells(a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y, v1x, v1y, v2x, v2y, g1x, g1y, g2x, g2y):
    # Define plot using matplotlib instead of pylab
    x1 = min(0, a1x + a2x, b1x + b2x, g1x + g2x, v1x + v2x, a1x, a2x, b1x, b2x, g1x, g2x, v1x, v2x)
    x2 = max(0, a1x + a2x, b1x + b2x, g1x + g2x, v1x + v2x, a1x, a2x, b1x, b2x, g1x, g2x, v1x, v2x)
    y1 = min(0, a1y + a2y, b1y + b2y, g1y + g2y, v1y + v2y, a1y, a2y, b1y, b2y, g1y, g2y, v1y, v2y)
    y2 = max(0, a1y + a2y, b1y + b2y, g1y + g2y, v1y + v2y, a1y, a2y, b1y, b2y, g1y, g2y, v1y, v2y)

    p1 = [0, 0]
    p2 = [v1x, v1y]
    p3 = [v1x + v2x, v1y + v2y]
    p4 = [v2x, v2y]
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', lw=0.5, color='green')
    plt.plot([p2[0], p3[0]], [p2[1], p3[1]], '-', lw=0.5, color='green')
    plt.plot([p3[0], p4[0]], [p3[1], p4[1]], '-', lw=0.5, color='green')
    plt.plot([p4[0], p1[0]], [p4[1], p1[1]], '-', lw=0.5, color='green')

    p1 = [0, 0]
    p2 = [g1x, g1y]
    p3 = [g1x + g2x, g1y + g2y]
    p4 = [g2x, g2y]
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', lw=1, color='black')
    plt.plot([p2[0], p3[0]], [p2[1], p3[1]], '-', lw=1, color='black')
    plt.plot([p3[0], p4[0]], [p3[1], p4[1]], '-', lw=1, color='black')
    plt.plot([p4[0], p1[0]], [p4[1], p1[1]], '-', lw=1, color='black')

    shift = 0.5
    p1 = [0 + shift, 0]
    p2 = [a1x + shift, a1y]
    p3 = [a1x + a2x + shift, a1y + a2y]
    p4 = [a2x + shift, a2y]
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', lw=1, color='red')
    plt.plot([p2[0], p3[0]], [p2[1], p3[1]], '-', lw=1, color='red')
    plt.plot([p3[0], p4[0]], [p3[1], p4[1]], '-', lw=1, color='red')
    plt.plot([p4[0], p1[0]], [p4[1], p1[1]], '-', lw=1, color='red')

    p1 = [0 + shift, 0]
    p2 = [b1x + shift, b1y]
    p3 = [b1x + b2x + shift, b1y + b2y]
    p4 = [b2x + shift, b2y]
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', lw=1, color='blue')
    plt.plot([p2[0], p3[0]], [p2[1], p3[1]], '--', lw=1, color='blue')
    plt.plot([p3[0], p4[0]], [p3[1], p4[1]], '--', lw=1, color='blue')
    plt.plot([p4[0], p1[0]], [p4[1], p1[1]], '--', lw=1, color='blue')

    dx = x2 - x1
    dy = y2 - y1
    more = 0.05
    if x1 < y1:
        y1 = x1
    else:
        x1 = y1
    if x2 > y2:
        y2 = x2
    else:
        x2 = y2
    if dx > dy:
        dy = dx
    else:
        dx = dy

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def floatize(sr):
    t = []
    for i in range(len(sr)):
        t.append(float(sr[i]))
    return t

def floatize_scale(sr, scale):
    t = []
    for i in range(len(sr)):
        t.append(scale * float(sr[i]))
    return t

def integerize(sr):
    t = []
    for i in range(len(sr)):
        t.append(int(sr[i]))
    return t

def cartesian_to_direct(cartesian_xyz, v1, v2, v3):
    c1, c2, c3 = np.linalg.solve([[v1[0], v2[0], v3[0]], [v1[1], v2[1], v3[1]], [v1[2], v2[2], v3[2]]], cartesian_xyz)
    return c1, c2, c3

def make_direct(atom_coords, v1, v2, v3):
    coords_direct = []
    for i in range(len(atom_coords)):
        c1, c2, c3 = cartesian_to_direct(atom_coords[i][0:3], v1, v2, v3)
        if len(atom_coords[i]) > 3:
            f1, f2, f3 = atom_coords[i][3:6]
            coords_direct.append([c1, c2, c3, f1, f2, f3])
        else:
            coords_direct.append([c1, c2, c3])
    return coords_direct

def make_cartesian(c1, c2, c3, a1, a2, a3):
    x = c1 * a1[0] + c2 * a2[0] + c3 * a3[0]
    y = c1 * a1[1] + c2 * a2[1] + c3 * a3[1]
    z = c1 * a1[2] + c2 * a2[2] + c3 * a3[2]
    return x, y, z

def write_file(name, a1x, a1y, a1z, a2x, a2y, a2z, a3x, a3y, a3z, atoms_x, selective_dynamics, chemical_symbols, atoms_coords_direct, comment, exchange, zfix):
    fl = open(name, 'w')
    fl.write(comment)
    t = " {:> 18.14f}".format(1.0) + ' \n'
    fl.write(t)
    if exchange == 0:
        t = '  {:> 21.16f} {:> 21.16f} {:> 21.16f}'.format(a1x, a1y, a1z) + '\n'
        fl.write(t)
        t = '  {:> 21.16f} {:> 21.16f} {:> 21.16f}'.format(a2x, a2y, a2z) + '\n'
        fl.write(t)
    else:
        t = '  {:> 21.16f} {:> 21.16f} {:> 21.16f}'.format(a2x, a2y, a2z) + '\n'
        fl.write(t)
        t = '  {:> 21.16f} {:> 21.16f} {:> 21.16f}'.format(a1x, a1y, a1z) + '\n'
        fl.write(t)
    t = '  {:> 21.16f} {:> 21.16f} {:> 21.16f}'.format(a3x, a3y, a3z) + '\n'
    fl.write(t)
    t = '   '
    if chemical_symbols != -1:
        for i in range(len(chemical_symbols)):
            t += chemical_symbols[i] + '   '
        t += ' \n'
        fl.write(t)
    t = '     '
    for i in range(len(atoms_x)):
        t += str(atoms_x[i]) + '   '
    t += ' \n'
    fl.write(t)

    if zfix != None:
        selective_dynamics = True

    if selective_dynamics:
        fl.write('Selective Dynamics\n')

    fl.write('Direct\n')
    for i in range(len(atoms_coords_direct)):
        if exchange == 0:
            t = ' {:> 19.16f} {:> 19.16f} {:> 19.16f}'.format(atoms_coords_direct[i][0], atoms_coords_direct[i][1], atoms_coords_direct[i][2])
        else:
            t = ' {:> 19.16f} {:> 19.16f} {:> 19.16f}'.format(atoms_coords_direct[i][1], atoms_coords_direct[i][0], atoms_coords_direct[i][2])

        z = atoms_coords_direct[i][2] * a3z
        if selective_dynamics:
            if zfix == None:
                t = t + ' T T T '
            else:
                if zfix <= 0:
                    if z > abs(zfix):
                        t = t + ' F F F '
                    else:
                        t = t + ' T T T '
                else:
                    if z < zfix:
                        t = t + ' F F F '
                    else:
                        t = t + ' T T T '
        t += '  \n'
        fl.write(t)
    fl.close()

def analyze_file(filename):
    fl = open(filename, 'r')
    tx = fl.readlines()
    fl.close()
    comment = tx[0]
    scale = float(tx[1].split()[0])
    ax, ay, az = floatize_scale(tx[2].split(), scale)
    bx, by, bz = floatize_scale(tx[3].split(), scale)
    cx, cy, cz = floatize_scale(tx[4].split(), scale)
    try:
        atoms_t = int(tx[5].split()[0])
        atoms_x = integerize(tx[5].split())
        chemsymbols = -1
        if tx[6][0] in ['s', 'S']:
            start = 8
            selective_dynamics = True
            if tx[7][0] in ['D', 'd']:
                type_of_coordinates = 'Direct'
            else:
                type_of_coordinates = 'Cartesian'
        else:
            selective_dynamics = False
            start = 7
            if tx[6][0] in ['D', 'd']:
                type_of_coordinates = 'Direct'
            else:
                type_of_coordinates = 'Cartesian'
    except:
        chemsymbols = tx[5].split()
        atoms_x = integerize(tx[6].split())
        if tx[7][0] in ['s', 'S']:
            selective_dynamics = True
            start = 9
            if tx[8][0] in ['D', 'd']:
                type_of_coordinates = 'Direct'
            else:
                type_of_coordinates = 'Cartesian'
        else:
            selective_dynamics = False
            start = 8
            if tx[7][0] in ['D', 'd']:
                type_of_coordinates = 'Direct'
            else:
                type_of_coordinates = 'Cartesian'
    atoms = 0
    for p in atoms_x:
        atoms += p

    atomic_coordinates = []
    for i in range(atoms):
        a, b, c = floatize(tx[start + i].split()[0:3])
        if selective_dynamics:
            f1, f2, f3 = tx[start + i].split()[3:6]
            atomic_coordinates.append([a, b, c, f1, f2, f3])
        else:
            atomic_coordinates.append([a, b, c])

    return ax, ay, az, bx, by, bz, cx, cy, cz, atoms_x, atoms, selective_dynamics, type_of_coordinates, chemsymbols, atomic_coordinates, comment


# Command line arguments and main logic
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', nargs='?', default='results.dat', help='Input file which contains results from the match_cells.py run')
parser.add_argument('index', type=int, default=-1, help='Index of the selected solution from the results file.')
parser.add_argument('--tolerance', type=int, default=1, help='Tolerance when searching for the atoms in the common cell.')
parser.add_argument('--tolerance_float', type=float, default=1e-4, help='Tolerance when searching for atoms in the new unit cell.')
parser.add_argument('--shift11', type=float, default=0.0, help='Shift of atoms in the first cell in direct coordinates along the first vector.')
parser.add_argument('--shift12', type=float, default=0.0, help='Shift of atoms in the first cell in direct coordinates along the second vector.')
parser.add_argument('--shift13', type=float, default=0.0, help='Shift of atoms in the first cell in direct coordinates along the third vector.')
parser.add_argument('--shift1x', type=float, default=0.0, help='Shift of atoms in the first cell in cartesian coordinates along x.')
parser.add_argument('--shift1y', type=float, default=0.0, help='Shift of atoms in the first cell in cartesian coordinates along y.')
parser.add_argument('--shift1z', type=float, default=0.0, help='Shift of atoms in the first cell in cartesian coordinates along z.')
parser.add_argument('--shift21', type=float, default=0.0, help='Shift of atoms in the second cell in direct coordinates along the first vector.')
parser.add_argument('--shift22', type=float, default=0.0, help='Shift of atoms in the second cell in direct coordinates along the second vector.')
parser.add_argument('--shift23', type=float, default=0.0, help='Shift of atoms in the second cell in direct coordinates along the third vector.')
parser.add_argument('--shift2x', type=float, default=0.0, help='Shift of atoms in the second cell in cartesian coordinates along x.')
parser.add_argument('--shift2y', type=float, default=0.0, help='Shift of atoms in the second cell in cartesian coordinates along y.')
parser.add_argument('--shift2z', type=float, default=0.0, help='Shift of atoms in the second cell in cartesian coordinates along z.')
parser.add_argument('--output', type=str, default='spc_def.vasp', help='Name of the output file with the resulting cell.')
parser.add_argument('--zfix', type=float, default=None, help='Z coordinate above which all atoms are free to relax, and below are fixed. If negative, atoms above abs(zfix) are fixed.')
parser.add_argument('--draw', type=int, default=0, help='Drawing a two unit cells and a combined one.')
args = parser.parse_args()

# Read the results.dat file to get the input file paths

def read_first_line_from_results(filename='results.dat'):
    with open(filename, 'r') as f:
        # Read only the first line
        first_line = f.readline()
    
    # Split the first line into parts and return the first two filenames
    parts = first_line.strip().split()
    if len(parts) >= 2:
        return parts[:2]  # Return the first two filenames
    else:
        return []  # Return an empty list if there aren't at least two filenames

# Get the selected file pair from the first line
selected_files = read_first_line_from_results()

print(selected_files)

# Check if there are two files in the first line and generate output filename
if len(selected_files) == 2:
    file1, file2 = selected_files
    base1, ext1 = os.path.splitext(file1)
    base2, ext2 = os.path.splitext(file2)
    
    if ext1 == ext2:  # Ensure both files have the same extension
        # Extract base name and indices (e.g., graph0_0, graph0_15)
        base_name1, index1 = base1.rsplit('_', 1)
        base_name2, index2 = base2.rsplit('_', 1)
        
        if base_name1 == base_name2:
            # Generate output filename based on the base name and indices
            output_filename = f"spc_{base_name1}_{index1}-{index2}{ext1}"
            print(f"Output file set to: {output_filename}")
        else:
            # Fallback output filename if base names or indices do not match
            args.output = f"spc_{base_name1}{ext1}"
else:
    print("Insufficient data in the first line.")


print ('\n=====================================================================================')
print ('             Cell match: Cell matching code for DFT or other calculations           ')
print ('=====================================================================================')
print ('Copyright (C) 2014 Predrag Lazic                                                 ')
print ('                         plazicx@gmail.com                                         ')
print ('Please visit https://sites.google.com/site/plazicx/codes                          ')
print ('=====================================================================================')
print ('                                                                                   ')
print ('                     Post-processing utility "generate_cell.py"                     ')
print ('      >>> Generates common unit cell from the results found with match_cells <<<    ')
print ('=====================================================================================')

# Defining input and output files
index = args.index
input_file = args.input_file
tolerance = args.tolerance
shift11 = args.shift11
shift12 = args.shift12
shift13 = args.shift13
shift1x = args.shift1x
shift1y = args.shift1y
shift1z = args.shift1z
shift21 = args.shift21
shift22 = args.shift22
shift23 = args.shift23
shift2x = args.shift2x
shift2y = args.shift2y
shift2z = args.shift2z
output_file = output_filename
zfix = args.zfix
draw = args.draw
tolerance_float = args.tolerance_float

# Reading input files and processing
fl = open(input_file, 'r')
tx = fl.readlines()
fl.close()

file1, file2 = tx[0].split()
selected = tx[4 + index].split()

total_atoms = int(selected[5])
ratio1 = int(selected[7])
ratio2 = int(selected[8])
i11, i12, i21, i22 = integerize(selected[10:14])
j11, j12, j21, j22 = integerize(selected[15:19])

print(file1, file2, index)
print(total_atoms, ratio1, ratio2, i11, i12, i21, i22, j11, j12, j21, j22)

print(f"Creating a new cell out of files {file1} and {file2} with total {total_atoms} atoms.")

# in principle your unit cell should be orthogonal so that c is perpendicular to the a-b plane
a1x, a1y, a1z, a2x, a2y, a2z, a3x, a3y, a3z, atoms_x1, atoms1, selective_dynamics1, type_of_coordinates1, chemical_symbols1, atoms1_coords, comment1 = analyze_file(file1)
b1x, b1y, b1z, b2x, b2y, b2z, b3x, b3y, b3z, atoms_x2, atoms2, selective_dynamics2, type_of_coordinates2, chemical_symbols2, atoms2_coords, comment2 = analyze_file(file2)

# notice than when printing out the files - scale will always be 1.000 and coordinates will always be Direct
if type_of_coordinates1 == 'Cartesian':
    atoms1_coords_direct = make_direct(atoms1_coords, [a1x, a1y, a1z], [a2x, a2y, a2z], [a3x, a3y, a3z])
else:
    atoms1_coords_direct = atoms1_coords

if type_of_coordinates2 == 'Cartesian':
    atoms2_coords_direct = make_direct(atoms2_coords, [b1x, b1y, b1z], [b2x, b2y, b2z], [b3x, b3y, b3z])
else:
    atoms2_coords_direct = atoms2_coords


coef1 = i11
coef2 = i12
coef3 = j11
coef4 = j12

v1nx = coef1 * a1x + coef2 * a2x
v1ny = coef1 * a1y + coef2 * a2y
v1nz = a1z
v2nx = coef3 * a1x + coef4 * a2x
v2ny = coef3 * a1y + coef4 * a2y
v2nz = a2z
v3nx = a3x
v3ny = a3y
v3nz = a3z

coef21 = i21
coef22 = i22
coef23 = j21
coef24 = j22

g1nx = coef21 * b1x + coef22 * b2x
g1ny = coef21 * b1y + coef22 * b2y
g1nz = b1z
g2nx = coef23 * b1x + coef24 * b2x
g2ny = coef23 * b1y + coef24 * b2y
g2nz = b2z
g3nx = b3x
g3ny = b3y
g3nz = b3z


volume = g1nx * g2ny - g1ny * g2nx
if volume < 0:
    exchange = 1
else:
    exchange = 0

print(g1nx, g2nx)
print(g2nx, g2ny)
print(f"Exchange cell vectors: {exchange}")

omjer1 = abs((v1nx * v2ny - v1ny * v2nx) / (a1x * a2y - a1y * a2x))
omjer2 = abs((g1nx * g2ny - g1ny * g2nx) / (b1x * b2y - b1y * b2x))

if draw == 1:
    draw_unit_cells(a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y, v1nx, v1ny, v2nx, v2ny, g1nx, g1ny, g2nx, g2ny)

atom_counter_1 = 0
atom_counter_2 = 0

pronadjeni_dupli_1 = 0
pronadjeni_dupli_2 = 0

tolerance_r = tolerance

atoms_already_in_1 = []
atoms_already_in_1_shifted = []

if copysign(1, coef1) != copysign(1, coef3):
    i1_1 = min(coef1, coef3) - tolerance_r
    i1_2 = max(coef1, coef3) + tolerance_r
else:
    i1_1 = min(coef1 + coef3, 0) - tolerance_r
    i1_2 = max(coef1 + coef3, 0) + tolerance_r

if copysign(1, coef2) != copysign(1, coef4):
    i2_1 = min(coef2, coef4) - tolerance_r
    i2_2 = max(coef2, coef4) + tolerance_r
else:
    i2_1 = min(coef2 + coef4, 0) - tolerance_r
    i2_2 = max(coef2 + coef4, 0) + tolerance_r

for p in range(atoms1):
    v1, v2, v3 = atoms1_coords[p][0:3]
    for i1 in range(i1_1, i1_2):
        for i2 in range(i2_1, i2_2):
            x, y, z = make_cartesian(i1 + v1, i2 + v2, v3, [a1x, a1y, a1z], [a2x, a2y, a2z], [a3x, a3y, a3z])
            c1, c2, c3 = cartesian_to_direct([x, y, z], [v1nx, v1ny, v1nz], [v2nx, v2ny, v2nz], [v3nx, v3ny, g3nz])
            dodani_atom = 0
            if c1 >= -tolerance_float and c1 <= 1 + tolerance_float:
                if c2 >= -tolerance_float and c2 <= 1 + tolerance_float:
                    noidentical = 1
                    for i in range(len(atoms_already_in_1)):
                        c1noidentical = 1
                        c2noidentical = 1
                        c3noidentical = 1
                        c1o, c2o, c3o = atoms_already_in_1[i]
                        if abs((c1o - c1) - round(c1o - c1)) < tolerance_float:
                            c1noidentical = 0
                        if abs((c2o - c2) - round(c2o - c2)) < tolerance_float:
                            c2noidentical = 0
                        if abs(c3o - c3) < tolerance_float:
                            c3noidentical = 0
                        if c1noidentical + c2noidentical + c3noidentical == 0:
                            noidentical = 0
                    if noidentical == 1:
                        # now we do the shifts!!
                        x, y, z = make_cartesian(c1, c2, c3, [v1nx, v1ny, g1nz], [v2nx, v2ny, g2nz], [v3nx, v3ny, g3nz])
                        x = x + shift1x + shift11 * a1x + shift12 * a2x + shift13 * a3x
                        y = y + shift1y + shift11 * a1y + shift12 * a2y + shift13 * a3y
                        z = z + shift1z + shift11 * a1z + shift12 * a2z + shift13 * a3z
                        c1s, c2s, c3s = cartesian_to_direct([x, y, z], [v1nx, v1ny, v1nz], [v2nx, v2ny, v2nz], [v3nx, v3ny, g3nz])
                        atoms_already_in_1.append([c1, c2, c3])
                        atoms_already_in_1_shifted.append([c1s, c2s, c3s])
                        atom_counter_1 += 1
                        dodani_atom = 1
                    else:
                        pronadjeni_dupli_1 += 1

# Continue for atoms in the second cell (atoms_already_in_2, etc.)
atoms_already_in_2 = []
atoms_already_in_2_shifted = []

if copysign(1, coef21) != copysign(1, coef23):
    i1_1 = min(coef21, coef23) - tolerance_r
    i1_2 = max(coef21, coef23) + tolerance_r
else:
    i1_1 = min(coef21 + coef23, 0) - tolerance_r
    i1_2 = max(coef21 + coef23, 0) + tolerance_r

if copysign(1, coef22) != copysign(1, coef24):
    i2_1 = min(coef22, coef24) - tolerance_r
    i2_2 = max(coef22, coef24) + tolerance_r
else:
    i2_1 = min(coef22 + coef24, 0) - tolerance_r
    i2_2 = max(coef22 + coef24, 0) + tolerance_r

for p in range(atoms2):
    v1, v2, v3 = atoms2_coords[p][0:3]
    for i1 in range(i1_1, i1_2):
        for i2 in range(i2_1, i2_2):
            x, y, z = make_cartesian(i1 + v1, i2 + v2, v3, [b1x, b1y, b1z], [b2x, b2y, b2z], [b3x, b3y, b3z])
            c1, c2, c3 = cartesian_to_direct([x, y, z], [g1nx, g1ny, g1nz], [g2nx, g2ny, g2nz], [g3nx, g3ny, g3nz])
            dodani_atom = 0
            if c1 >= -tolerance_float and c1 <= 1 + tolerance_float:
                if c2 >= -tolerance_float and c2 <= 1 + tolerance_float:
                    noidentical = 1
                    for i in range(len(atoms_already_in_2)):
                        c1noidentical = 1
                        c2noidentical = 1
                        c3noidentical = 1
                        c1o, c2o, c3o = atoms_already_in_2[i]
                        if abs((c1o - c1) - round(c1o - c1)) < tolerance_float:
                            c1noidentical = 0
                        if abs((c2o - c2) - round(c2o - c2)) < tolerance_float:
                            c2noidentical = 0
                        if abs(c3o - c3) < tolerance_float:
                            c3noidentical = 0
                        if c1noidentical + c2noidentical + c3noidentical == 0:
                            noidentical = 0
                    if noidentical == 1:
                        # now we do the shifts!!
                        x, y, z = make_cartesian(c1, c2, c3, [g1nx, g1ny, g1nz], [g2nx, g2ny, g2nz], [g3nx, g3ny, g3nz])
                        x = x + shift2x + shift21 * b1x + shift22 * b2x + shift23 * b3x
                        y = y + shift2y + shift21 * b1y + shift22 * b2y + shift23 * b3y
                        z = z + shift2z + shift21 * b1z + shift22 * b2z + shift23 * b3z
                        c1s, c2s, c3s = cartesian_to_direct([x, y, z], [g1nx, g1ny, g1nz], [g2nx, g2ny, g2nz], [g3nx, g3ny, g3nz])
                        atoms_already_in_2.append([c1, c2, c3])
                        atoms_already_in_2_shifted.append([c1s, c2s, c3s])
                        atom_counter_2 += 1
                        dodani_atom = 1
                    else:
                        pronadjeni_dupli_2 += 1

print(f"After creation of the new cell - the atoms from the first cell should be {atoms1 * ratio1} and there are {atom_counter_1}")
check = 1
if atoms1 * ratio1 == atom_counter_1:
    print('OK')
else:
    print("Problem! Try increasing tolerance --tolerance 5 for example")
    check = 0

print(f"After creation of the new cell - the atoms from the second cell should be {atoms2 * ratio2} and there are {atom_counter_2}")
if atoms2 * ratio2 == atom_counter_2:
    print('OK')
else:
    print("Problem! Try increasing tolerance --tolerance 5 for example")
    check = 0

# Second cell is the one that remains unchanged
if check == 1:
    print(f"Everything seems OK - generating output file {output_file}")
    chemical_symbols_new = []
    unique_chemical_symbols = 1
    if chemical_symbols1 != -1:
        for p in chemical_symbols1:
            chemical_symbols_new.append(p)
    if chemical_symbols2 != -1:
        for p in chemical_symbols2:
            if p in chemical_symbols1:
                unique_chemical_symbols = 0
            chemical_symbols_new.append(p)
    atoms_new = []
    for p in atoms_x1:
        atoms_new.append(p * ratio1)
    for p in atoms_x2:
        atoms_new.append(p * ratio2)
    comment_new = comment1[0:-1] + comment2[0:-1] + ' \n'
    atomsnew_coords_direct = []
    # Notice we want the shifted atoms from the first cell
    for atom in atoms_already_in_1_shifted:
        atomsnew_coords_direct.append(atom)
    for atom in atoms_already_in_2_shifted:
        atomsnew_coords_direct.append(atom)

    selective_dynamics_new = False
    if unique_chemical_symbols == 1:
        write_file(output_file, g1nx, g1ny, g1nz, g2nx, g2ny, g2nz, g3nx, g3ny, g3nz, atoms_new, selective_dynamics_new, chemical_symbols_new, atomsnew_coords_direct, comment_new, exchange, zfix)
    else:
        print("REARRANGE POSCAR")
        write_file('POSCAR_temp', g1nx, g1ny, g1nz, g2nx, g2ny, g2nz, g3nx, g3ny, g3nz, atoms_new, selective_dynamics_new, chemical_symbols_new, atomsnew_coords_direct, comment_new, exchange, zfix)
        a1x, a1y, a1z, a2x, a2y, a2z, a3x, a3y, a3z, atoms_x1, atoms1, selective_dynamics1, type_of_coordinates1, chemical_symbols1, atoms1_coords, comment1 = analyze_file('POSCAR_temp')
        chemistry = []
        for ch in chemical_symbols1:
            if ch not in chemistry:
                chemistry.append(ch)
        lista = []
        for p in range(len(chemistry)):
            lista.append([])
        counter = 0
        for p in range(len(atoms_x1)):
            for j in range(atoms_x1[p]):
                lista[chemistry.index(chemical_symbols1[p])].append(counter)
                counter += 1
        atoms_xnew = []
        for p in range(len(lista)):
            atoms_xnew.append(len(lista[p]))
        atoms1_new_coords = []
        for p in range(len(lista)):
            for k in range(len(lista[p])):
                atoms1_new_coords.append(atoms1_coords[lista[p][k]])
        comment_new = ''
        for p in chemistry:
            comment_new += f' {p}'
        comment_new += ' \n'
        write_file(output_file, g1nx, g1ny, g1nz, g2nx, g2ny, g2nz, g3nx, g3ny, g3nz, atoms_xnew, selective_dynamics_new, chemistry, atoms1_new_coords, comment_new, exchange, zfix)
        os.system('rm POSCAR_temp')
else:
    print("Something went wrong, doing nothing!")