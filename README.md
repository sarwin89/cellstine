# Moiré Supercell Finder & Generator

This repository provides a pair of Python tools for identifying and
building commensurate supercells between two two-dimensional lattices
(e.g. twisted bilayers).

There are two command-line programs:

* `finder.py` – search a range of twist angles and report candidate
  supercells that require minimal linear strain.
* `generator.py` – take a results file produced by the finder and
  construct a merged POSCAR containing both layers in the common cell.

Under the hood the computation is organised into the `moire` package
so that the core routines can be reused by a web service or other
front-end.

## Features

* fast vectorised algebra with NumPy
* optional filtering by strain, atom count, twist angle
* modular design with separate I/O and geometry modules
* results file records input filenames so generator is automatic
* simple generator algorithm that wraps atomic positions into the
  new lattice

The original MATLAB-style code that inspired this project is kept in
`Reference/` for comparison.

## Quick start

Use the Python interpreter installed in your environment – e.g.:  

```powershell
& "C:/Users/<you>/AppData/Local/Python/pythoncore-3.14-64/python.exe" \
    finder.py graph.vasp mos2.vasp 0 5 --angle_step 0.1 \
    --nindex 8 --tolerance 1e-3 --lin_tol 5e-3
```

A `results.dat` file will be written; the first line contains the two
input filenames.  Pick a row index and generate the supercell:

```powershell
& "...\python.exe" generator.py results.dat 1 --output spc.vasp
```

## Developer documentation

Each module in `moire/` contains extensive docstrings.  You can import
and call the functions directly if you are writing another program or
a web service.

The package has the following structure:

* `moire/io.py` – POSCAR parsing & writing
* `moire/lattice.py` – geometric routines (rotate, strain, span,
  supercell construction)
* `moire/finder.py` – search algorithm and CLI
* `moire/generator.py` – build combined cell and CLI

Plugging the package into a web front-end simply requires invoking the
`find_supercells` and `build_supercell` functions; the command-line
scripts are just thin wrappers.

## Tests & examples

A few example POSCARs (`graph.vasp`, `mos2.vasp`) are included; run
the finder with them to verify results.  You can also create your own
results files for known twist angles to check correctness.

## Licensing & attribution

This code is distributed under the MIT license (you can choose any
appropriate open-source license for your project).  Portions of the
algorithm derive from the `CellMatch` code by Predrag Lazic; see
`Reference/` for the original implementation and maintainers.

## Future work

Planned enhancements include:

* improved generator that handles z-fixing, selective dynamics, and
  atom merging rules
* multiprocessing in the finder
* Python package installation with `setup.py` / `pyproject.toml`
* web front-end (separate repository)

Contributions welcome!
