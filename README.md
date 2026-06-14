# CELLSTINE

**CELL Superlattice Transformation INterface and Engine**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](#installation)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version 4.0.0](https://img.shields.io/badge/version-4.0.0-informational)](pyproject.toml)

CELLSTINE is a Python package and guided CLI for building VASP-style atomistic structures: moire supercells, molecule-on-substrate systems, surfaces, interfaces, defects, and symmetry-reduced cells.

## What It Does

- Finds and builds commensurate bilayer and N-layer moire structures.
- Places, moves, rotates, and reframes molecules on substrate slabs.
- Generates slabs from bulk cells and detects adsorption sites.
- Builds slab-on-slab interfaces and screens bulk surface matches.
- Analyses inequivalent defect sites and generates defect POSCARs.
- Performs symmetry analysis and primitive/conventional cell reduction.
- Writes manifests in `runs/` so workflow steps can be reproduced.

## Installation

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[symmetry]"
python -m pip install -e ".[pymatgen]"
python -m pip install -e ".[viz]"
python -m pip install -e ".[plotly]"
python -m pip install -e ".[all]"
```

Check the CLI:

```bash
cellstine --help
cellstine --version
```

## Quick Start

Launch the guided interface:

```bash
cellstine
```

Or run a workflow directly:

```bash
cellstine moire find input/examples/mos2.vasp input/examples/mos2.vasp --nindex 8
cellstine interface surface input/examples/Au_Bulk.vasp --miller 111 --layers 4 --vacuum 15
cellstine interface sites output/examples/Au_Bulk_111_surface.vasp
cellstine symmetry analyse input/examples/Au_Bulk.vasp
```

## Public Examples

Small, tracked examples live in:

```text
input/examples/    Source structures for demos and tests
output/examples/   Curated example outputs
```

Normal local work still uses:

```text
input/    Your source structures
output/   Generated structures, plots, and reports
runs/     Manifests and intermediate workflow artifacts
```

Generated files outside the `examples/` folders are ignored by Git.

## Documentation

- [Documentation index](docs/README.md)
- [Quickstart](docs/quickstart.md)
- [CLI guide](docs/cli.md)
- [Moire workflow](docs/workflows/moire.md)
- [Adsorbate workflow](docs/workflows/adsorbate.md)
- [Interface workflow](docs/workflows/interface.md)
- [Defect workflow](docs/workflows/defect.md)
- [Symmetry workflow](docs/workflows/symmetry.md)
- [Architecture notes](docs/architecture.md)
- [Moire search performance](docs/moire-performance.md)
- [Full usage guide](USAGE_GUIDE.md)
- [Roadmap](ROADMAP.md)

## Development

Run the test suite:

```bash
python -m unittest discover -s tests -q
```

The package uses a `src/` layout, with workflow code under `src/cellstine/` and regression tests under `tests/`.

## Author

Made by Sarwin Chandran, 2026.

## License

CELLSTINE is released under the [MIT License](LICENSE).
