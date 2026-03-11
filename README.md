# Moire Supercell Finder

Fast, vectorized tools for finding commensurate moire supercells from VASP POSCAR files.

The project is split into a reusable Python package and thin top-level CLI entry points:

- `moire/`: all core functionality
- `cellfind.py`: fast commensurate-angle shortlist CLI
- `finder.py`: strain-aware supercell search CLI
- `generator.py`: exact supercell builder CLI

The legacy research scripts are preserved in `Reference/` for comparison.

## Features

- Vectorized NumPy span matching for speed
- Fast commensurate-angle search based on equal-length lattice spans
- Strain-aware supercell search with layer-wise and average strain metrics
- Exact supercell reconstruction from integer match coefficients
- POSCAR read/write support for Direct, Cartesian, species, and Selective Dynamics
- Generator support for layer shifts and optional `zfix`
- Reference-backed tests for MoS2/MoS2 commensurate structures

## Repository layout

```text
.
+-- cellfind.py          # CLI wrapper for moire.angles
+-- finder.py            # CLI wrapper for moire.finder
+-- generator.py         # CLI wrapper for moire.generator
+-- moire/
¦   +-- __init__.py
¦   +-- angles.py        # commensurate-angle search
¦   +-- finder.py        # strain-aware candidate search
¦   +-- generator.py     # exact supercell generation
¦   +-- io.py            # POSCAR parsing/writing
¦   +-- lattice.py       # vectorized lattice math
+-- Reference/           # older working/reference scripts
+-- Results/             # saved reference MoS2/MoS2 supercells
+-- tests/
```

## Recommended workflow

### 1. Shortlist commensurate angles

Use `cellfind.py` to identify likely twist angles before running the heavier full search.

```powershell
& 'C:\Users\Sarwi\AppData\Local\Python\pythoncore-3.14-64\python.exe' cellfind.py mos2.vasp mos2.vasp 12 --strain_tolerance 0.002 --max_angle 30
```

This prints a table of candidate angles with matching span coefficients.

### 2. Run the full supercell finder

Use `finder.py` on either an angle range, a fixed list of angles, or angles shortlisted by `cellfind`.

```powershell
& 'C:\Users\Sarwi\AppData\Local\Python\pythoncore-3.14-64\python.exe' finder.py mos2.vasp mos2.vasp --angles 13.15,21.787,27.9 --nindex 12 --tolerance 0.002 --lin_tol 0.002 --vector_strain_tol 0.002 --max_atoms 200
```

Or let the finder call the angle shortlisting stage internally:

```powershell
& 'C:\Users\Sarwi\AppData\Local\Python\pythoncore-3.14-64\python.exe' finder.py mos2.vasp mos2.vasp 0 30 --use_cellfind --nindex 12 --tolerance 0.002 --lin_tol 0.002 --angle_strain_tolerance 0.002 --vector_strain_tol 0.002 --max_atoms 200
```

The finder writes `results.dat` with:

- twist angle
- average strain
- layer-1 and layer-2 strain estimates
- total atoms
- supercell ratios
- integer coefficients for both matched layers
- vector matching errors

### 3. Generate the exact supercell

```powershell
& 'C:\Users\Sarwi\AppData\Local\Python\pythoncore-3.14-64\python.exe' generator.py results.dat 1 --output supercell.vasp
```

Useful generator options:

- `--preserve_layer 1|2|avg`
- `--shift11`, `--shift12`, `--shift13`
- `--shift1x`, `--shift1y`, `--shift1z`
- `--shift21`, `--shift22`, `--shift23`
- `--shift2x`, `--shift2y`, `--shift2z`
- `--zfix`

## Python API

### Angle shortlist

```python
from moire import angles, io

structure = io.read_poscar('mos2.vasp')
candidates = angles.find_commensurate_angles(
    structure.lattice,
    structure.lattice,
    nindex=12,
    strain_tolerance=2e-3,
    min_angle=0.0,
    max_angle=30.0,
)
```

### Full finder

```python
from moire import finder, io

structure = io.read_poscar('mos2.vasp')
results = finder.find_supercells(
    structure.lattice,
    structure.lattice,
    None,
    None,
    angles=[13.15, 21.787, 27.9],
    nindex=12,
    tol=2e-3,
    lin_tol=2e-3,
    vector_strain_tol=2e-3,
    atom_count1=structure.natoms,
    atom_count2=structure.natoms,
)
```

### Generator

```python
from moire import generator

lattice, positions_direct, counts, species, flags = generator.build_supercell(
    'mos2.vasp',
    'mos2.vasp',
    record,
)
```

## Validation

The current implementation is tested against the saved MoS2/MoS2 reference supercells in `Results/`.

Verified reference families:

- around `13.15 deg` -> `114` atoms
- around `21.787 deg` -> `42` atoms
- around `27.9 deg` -> `78` atoms

The generated supercells reproduce the same species counts and in-plane lattice lengths as the saved reference POSCARs.

## Tests

Run the test suite with:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\Sarwi\AppData\Local\Python\pythoncore-3.14-64\python.exe' -m unittest discover -s tests -q
```

## Notes

- `pos1` is the rotated layer in the finder flow.
- `pos2` is preserved by default in the generator flow.
- `Reference/` is kept intentionally for provenance and comparison.
