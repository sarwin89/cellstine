"""Core functionality for the moire supercell finder/generator.

This package provides the computational routines and command-line
interfaces used by the top-level ``finder.py`` and ``generator.py``
scripts.  It is organised to keep the mathematics and I/O separated,
making it easier to re-use the core logic in a web service or other
front-end.

Developers:
* ``moire.io`` handles reading and writing POSCAR-style files.
* ``moire.lattice`` contains geometric utilities (rotation, strain,
  vector spanning, etc.).
* ``moire.finder`` implements the search over twist angles and
  superlattice construction.
* ``moire.generator`` builds merged supercells from a results table.

User-facing entry points are provided by ``../finder.py`` and
``../generator.py`` at the repository root.

"""

__version__ = "0.1.0"
