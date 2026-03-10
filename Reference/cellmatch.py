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
import sys, time

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
    c1, c2, c3 = np.linalg.solve([[v1[0], v2[0], v3[0]], 
                                   [v1[1], v2[1], v3[1]], 
                                   [v1[2], v2[2], v3[2]]], cartesian_xyz)
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

def calculate_strain(a1, b1, c1, a2, b2, c2):
    a1x, a1y, a1z = a1
    b1x, b1y, b1z = b1
    c1x, c1y, c1z = c1
    a2x, a2y, a2z = a2
    b2x, b2y, b2z = b2
    c2x, c2y, c2z = c2
    
    # we put c2z and c1z here to 1.0 just to be sure, just in case the user used different z dimensions
    c1z = 1.0
    c2z = 1.0
    
    metric_tensor1 = np.matrix([[a1x**2 + a1y**2 + a1z**2, a1x*b1x + a1y*b1y + a1z*b1z, a1x*c1x + a1y*c1y + a1z*c1z],
                                [a1x*b1x + a1y*b1y + a1z*b1z, b1x**2 + b1y**2 + b1z**2, b1x*c1x + b1y*c1y + b1z*c1z],
                                [a1x*c1x + a1y*c1y + a1z*c1z, c1x*b1x + c1y*b1y + c1z*b1z, c1x**2 + c1y**2 + c1z**2]])
    
    metric_tensor2 = np.matrix([[a2x**2 + a2y**2 + a2z**2, a2x*b2x + a2y*b2y + a2z*b2z, a2x*c2x + a2y*c2y + a2z*c2z],
                                [a2x*b2x + a2y*b2y + a2z*b2z, b2x**2 + b2y**2 + b2z**2, b2x*c2x + b2y*c2y + b2z*c2z],
                                [a2x*c2x + a2y*c2y + a2z*c2z, c2x*b2x + c2y*b2y + c2z*b2z, c2x**2 + c2y**2 + c2z**2]])
    
    rt1 = np.linalg.cholesky(metric_tensor1).transpose()
    rt2 = np.linalg.cholesky(metric_tensor2).transpose()
    
    unit_matrix = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    
    evec = rt2 * rt1.getI() - unit_matrix
    strain1 = 0.5 * (evec + evec.transpose())
    strain2 = 0.5 * (evec + evec.transpose() + evec * evec.transpose())
    
    str1, str2, str3 = np.linalg.eig(strain2)[0]
    deformation = sqrt(str1**2 + str2**2 + str3**2) / 3.
    
    return deformation

def write_file(name, a1x, a1y, a1z, a2x, a2y, a2z, a3x, a3y, a3z, atoms_x, selective_dynamics, chemical_symbols, atoms_coords_direct, comment):
    fl = open(name, 'w')
    fl.write(comment)
    t = " {:> 18.14f}".format(1.0) + ' \n'
    fl.write(t)
    t = '  {:> 21.16f} {:> 21.16f} {:> 21.16f}'.format(a1x, a1y, a1z) + '\n'
    fl.write(t)
    t = '  {:> 21.16f} {:> 21.16f} {:> 21.16f}'.format(a2x, a2y, a2z) + '\n'
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
    if selective_dynamics:
        fl.write('Selective Dynamics\n')
    fl.write('Direct\n')
    for i in range(len(atoms_coords_direct)):
        t = ' {:> 19.16f} {:> 19.16f} {:> 19.16f}'.format(atoms_coords_direct[i][0], atoms_coords_direct[i][1], atoms_coords_direct[i][2])
        if len(atoms_coords_direct[i]) > 3:
            for j in range(3, 6):
                t += ' ' + atoms_coords_direct[i][j]
        t += '  \n'
        fl.write(t)
    fl.close()

def analyze_file(filename):
    fl = open(filename, 'r')
    tx = fl.readlines()
    fl.close()
    # we follow VASP file structure, first line is a comment
    # second line is a scaling factor or volume (if negative)
    # volume feature not supported
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


parser = argparse.ArgumentParser()
parser.add_argument('input_file1', nargs='?', default=None, help='input file for the first cell - the one that can also be rotated')
parser.add_argument('input_file2', nargs='?', default=None, help='input file for the second cell - the one which will be preserved in combined cell')
parser.add_argument('--nindex', type=int, default=10, help='Index of repetitions in space from -N to N.')
parser.add_argument('--tolerance', type=float, default=0.01, help='Tolerance when searching for common points in plane.')
parser.add_argument('--rotate', type=float, default=0.0, help='Angle of rotation for cell1 - in degrees.')
parser.add_argument('--maxatoms', type=int, default=-1, help='Do not show results for number of atoms larger than maxatoms.')
parser.add_argument('--maxstrain', type=float, default=-1, help='Do not show results for strain value larger than maxstrain.')
parser.add_argument('--unique', type=int, default=1, help='.')
parser.add_argument('--output', type=str, default='results.dat', help='Name of the output file with results.')
parser.add_argument('--show_progress', type=int, default=1, help='To show progress in percent.')
args = parser.parse_args()

print('                                                                                      \n'
      '===================================================================================== \n'
      '             cellmatch: Cell matching code for DFT or other calculations              \n'
      '===================================================================================== \n'
      'Copyright (C) 2025       Predrag Lazic & Sarwin Chandran                              \n'
      '                                                                                      \n'
      '===================================================================================== \n'
      '                                                                                      \n'
      '                     Post-processing utility "cellgen.py"                             \n'
      '            >>> Generates common unit cell from the results found here <<<            \n'
      '                                                                                      \n')

indent = '    '
############################################################################################################


# Defining input and output files
file1 = args.input_file1
file2 = args.input_file2

nindex = args.nindex
angle = args.rotate
maxatoms = args.maxatoms
maxstrain = args.maxstrain
unique = args.unique
linear_tolerance = args.tolerance
show_percent = args.show_progress
output_file = args.output

print(file1, file2, nindex)
print("Trying to find common cell for unit cells from files ", file1, " and ", file2)

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

# rotating the first cell (even if the angle is zero)

r1 = sqrt(a1x**2 + a1y**2)
a1 = atan2(a1y, a1x)

a1x_rotated = r1 * cos(a1 + angle * pi / 180.)
a1y_rotated = r1 * sin(a1 + angle * pi / 180.)

r2 = sqrt(a2x**2 + a2y**2)
a2 = atan2(a2y, a2x)

a2x_rotated = r2 * cos(a2 + angle * pi / 180.)
a2y_rotated = r2 * sin(a2 + angle * pi / 180.)

# we write out files for cell combinations - all coordinates are direct, no scaling factors

name1 = file1
name2 = file2

# now we can switch to rotated unit vectors

a1x = a1x_rotated
a1y = a1y_rotated

a2x = a2x_rotated
a2y = a2y_rotated

# now we are starting the search for points in space that coincide within the tolerance

distances = []
relative_distances = []
total_numbers = (2 * nindex + 1) ** 4
counter = 0

print("Searching for coincident points.")
for i1 in range(-nindex, nindex + 1):
    for i2 in range(-nindex, nindex + 1):
        if abs(i1) + abs(i2) != 0:
            for j1 in range(-nindex, nindex + 1):
                for j2 in range(-nindex, nindex + 1):
                    if abs(j1) + abs(j2) != 0:
                        counter += 1
                        v1x = a1x * i1 + a2x * i2
                        v1y = a1y * i1 + a2y * i2
                        v2x = b1x * j1 + b2x * j2
                        v2y = b1y * j1 + b2y * j2
                        epsilon = sqrt((v1x - v2x) ** 2 + (v1y - v2y) ** 2)
                        distances.append([epsilon, [i1, i2, j1, j2]])
                        relative_distances.append([epsilon / (sqrt(v1x ** 2 + v1y ** 2) + sqrt(v2x ** 2 + v2y ** 2)), [i1, i2, j1, j2]])
                        percent = counter / float(total_numbers) * 100
                        if show_percent == 1:
                            sys.stdout.write("\r%d%%" % percent)
                            sys.stdout.flush()

if show_percent == 1:
    sys.stdout.write("\r%d%%" % 100)
    sys.stdout.flush()
print(" Done!")

relative_distances.sort()

po_povrsini = []
po_povrsini_unique_omjer = []
po_povrsini_unique = []

found = 0
k = 0
while found == 0:
    if relative_distances[k][0] > linear_tolerance:
        found = 1
    else:
        k += 1

print("Found ", k, " vector candidates to build cells.")

# this is the definition of "zero"
zero_tolerance = 0.1

print("Now searching for common cells within the selected vector combinations.")

total = (k ** 2) / 2.

counter = 0
for i in range(k):
    for j in range(i, k):
        omjer1 = 0
        omjer2 = 0
        eps1, iovi1 = relative_distances[i]
        i11, i12, j11, j12 = iovi1
        eps2, iovi2 = relative_distances[j]
        i21, i22, j21, j22 = iovi2
        v1x = a1x * i11 + a2x * i12
        v1y = a1y * i11 + a2y * i12
        v2x = a1x * i21 + a2x * i22
        v2y = a1y * i21 + a2y * i22
        if v1x * v2y - v1y * v2x != 0:
            omjer1 = round(abs((v1x * v2y - v1y * v2x) / (a1x * a2y - a1y * a2x)))

        g1x = b1x * j11 + b2x * j12
        g1y = b1y * j11 + b2y * j12
        g2x = b1x * j21 + b2x * j22
        g2y = b1y * j21 + b2y * j22

        if g1x * g2y - g1y * g2x != 0:
            omjer2 = round(abs((g1x * g2y - g1y * g2x) / (b1x * b2y - b1y * b2x)))

        if omjer1 > zero_tolerance and omjer2 > zero_tolerance:
            surf1 = abs(v1x * v2y - v1y * v2x)
            surf2 = abs(g1x * g2y - g1y * g2x)
            strain = calculate_strain([v1x, v1y, 0], [v2x, v2y, 0], [0, 0, 1], [g1x, g1y, 0], [g2x, g2y, 0], [0, 0, 1])
            total_atoms = round(atoms1 * omjer1 + atoms2 * omjer2)
            length = sqrt(v1x ** 2 + v1y ** 2) * sqrt(v2x ** 2 + v2y ** 2)
            po_povrsini.append([strain, [omjer1, omjer2, length, total_atoms, [iovi1, iovi2], eps1, eps2]])

        counter += 1
        percent = counter / float(total) * 100
        if show_percent == 1:
            sys.stdout.write("\r%d%%" % percent)
            sys.stdout.flush()

if show_percent == 1:
    sys.stdout.write("\r%d%%" % 100)
    sys.stdout.flush()

print(" Done")

# now sorting by the strain

po_povrsini.sort()

print("Found  ", len(po_povrsini), " candidates for the new common cell.")
print("Analyzing.....")

po_povrsini_unique_strain = []

check_for_unique = unique

# numbers closer than this are considered identical (strain values)
tolerance_strain = 1e-4
tolerance_ratio = 1e-5

for i in range(len(po_povrsini)):
    percent = i / float(len(po_povrsini)) * 100
    if show_percent == 1:
        sys.stdout.write("\r%d%%" % percent)
        sys.stdout.flush()

    strain = po_povrsini[i][0]
    omjer1 = po_povrsini[i][1][0]
    omjer2 = po_povrsini[i][1][1]
    length = po_povrsini[i][1][2]
    found = 0
    if check_for_unique != 0:
        for j in range(len(po_povrsini_unique_strain)):
            strain_in = po_povrsini_unique_strain[j][0]
            omjer1_in, omjer2_in, length_in = po_povrsini_unique_strain[j][1]
            if abs(strain - strain_in) < tolerance_strain and abs(omjer1 / omjer2 - omjer1_in / omjer2_in) < tolerance_ratio:
                found = 1
    if found == 0:
        # now we are going to add it - but moreover - we shall add the one with the smallest surface, and smallest length of unit cell vectors!
        najmanji_index = i
        najmanji_omjer = omjer1
        najmanji_length = length
        brojalnik = i + 1
        exit = 0
        while exit == 0 and brojalnik < len(po_povrsini) and check_for_unique == 1:
            k = brojalnik
            strain_in = po_povrsini[k][0]
            omjer1_in, omjer2_in, length_in = po_povrsini[k][1][0:3]
            if abs(strain - strain_in) < tolerance_strain and abs(omjer1 / omjer2 - omjer1_in / omjer2_in) < tolerance_ratio:
                # identical strain and surface ratio - now we are checking for length
                if length_in < najmanji_length:
                    najmanji_index = k
                    najmanji_length = length_in
            brojalnik += 1
            # since they are sorted by strain - after this is satisfied we don't have to search any further
            if strain_in > strain + tolerance_strain:
                exit = 1
        po_povrsini_unique_strain.append([po_povrsini[najmanji_index][0], po_povrsini[najmanji_index][1][0:3]])
        po_povrsini_unique.append(po_povrsini[najmanji_index])

if show_percent == 1:
    sys.stdout.write("\r%d%%" % 100)
    sys.stdout.flush()

print(" Done.")

print("Writing results in the file : ", output_file)

fl = open(output_file, 'w')

print("------------------------------------------------------- RESULTS ---------------------------------------------")
print("-------------------------------------------------------------------------------------------------------------")
print("|  index  |       strain       |    atoms  |  surf_ratio  |         indices1      |         indices2        |")
print("-------------------------------------------------------------------------------------------------------------")

t = name1 + '      ' + name2 + ' \n'
fl.write(t)

fl.write("------------------------------------------------------- RESULTS ---------------------------------------------" + ' \n')
fl.write("--------------------------------------------------------------- ---------------------------------------------" + ' \n')
fl.write("|  index  |       strain       |    atoms  |  surf_ratio  |         indices1      |         indices2        |" + ' \n')
fl.write("-------------------------------------------------------------------------------------------------------------" + ' \n')

counter = 0
found = 0

counter_write = 1
last_number_of_atoms = -1
while found == 0 and counter < len(po_povrsini_unique):
    if po_povrsini_unique[counter][0] > maxstrain and maxstrain > 0:
        found = 1
    else:
        if po_povrsini_unique[counter][1][3] <= maxatoms or maxatoms < 0:
            t = '  {:> 21.16f} {:> 21.16f} {:> 21.16f}'.format(a1x, a1y, a1z) + '\n'
            strain = po_povrsini_unique[counter][0]
            natoms = int(po_povrsini_unique[counter][1][3])
            if last_number_of_atoms == -1:
                last_number_of_atoms = natoms
            omjer1, omjer2 = po_povrsini_unique[counter][1][0:2]
            omjer1 = int(omjer1)
            omjer2 = int(omjer2)
            i11, i12, i21, i22 = po_povrsini_unique[counter][1][4][0]
            j11, j12, j21, j22 = po_povrsini_unique[counter][1][4][1]
            writeout = 1
            if counter_write > 1 and unique == 1:
                if natoms >= last_number_of_atoms:
                    writeout = 0
                else:
                    last_number_of_atoms = natoms
            if writeout == 1:
                t = '|{:> 7d}  |  {:> 16.8f}  |  {:> 7d}  |   {:> 4d} {:> 4d}  |  {:> 4d} {:> 4d} {:> 4d} {:> 4d}  |  {:> 4d} {:> 4d} {:> 4d} {:> 4d}    |'.format(counter_write, strain, natoms, omjer1, omjer2, i11, i12, i21, i22, j11, j12, j21, j22)
                print(t)
                fl.write(t + ' \n')
                counter_write += 1
    counter += 1

print("-------------------------------------------------------------------------------------------------------------")
fl.write("-------------------------------------------------------------------------------------------------------------" + ' \n')

print(" Done. ")
print("")
print(" To really generate cells - use the cellgen.py script with the given results file ", output_file, " and selected solution index.")
if output_file == 'results.dat':
    print(" For example python cellgen.py 3 to get a cells defined by the third solution in the file.")
else:
    print(" For example python cellgen.py 3 --input_file ", output_file, " to get a cells defined by the third solution in the file.")
print(" ")
if counter_write == 1:
    print("The code did not find any matching cells. Try increasing the tolerance - say 0.02, or try to rotate the first cell by some angle.")

fl.close()

