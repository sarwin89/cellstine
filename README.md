# CELLSTINE

**CELL Superlattice Transformation INterface and Engine**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](#installation)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version 3.0.1](https://img.shields.io/badge/version-3.0.1-informational)](pyproject.toml)

CELLSTINE is a Python package and guided CLI for building VASP-style atomistic structures: moire supercells, molecule-on-substrate systems, surfaces, interfaces, defects, and symmetry-reduced cells.

## What It Does

- Finds and builds commensurate bilayer moire structures with the native Gram-form engine.
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
python -m pip install -e ".[cli]"
python -m pip install -e ".[all]"
```

Check the CLI:

```bash
cellstine --help
cellstine --version
```

When working from a checkout without installing the console script, use the
repo-local launcher:

```bash
python cellstine.py --help
```

## Quick Start

Launch the guided interface:

```bash
cellstine
```

Install `.[cli]` for the recommended Rich/Typer guided presentation. Base
installs remain dependency-free and automatically use the plain prompt UI; pass
`--plain` to force that fallback explicitly.

Or run a workflow directly:

```bash
cellstine moire search input/examples/mos2.vasp input/examples/mos2.vasp --length 20 --strain 0.01
cellstine moire build runs/moire/<run-id>/results.json --indexes 1 --interlayer-distance 3.35
cellstine surface build input/examples/Au_Bulk.vasp --miller 111 --layers 4 --vacuum 15
cellstine surface sites output/examples/Au_Bulk_111_surface.vasp
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

The native moire search writes schema-versioned `results.json`; previews,
builders, and visualizers consume that file rather than positional tables.
Here **strain** means principal logarithmic strain, `h = log(lambda)`, for a
principal stretch `lambda`. The accepted relative principal strain is bounded
by the sum of the top and bottom strain budgets and is shared optimally between
the layers for each candidate. This name is deliberate: it is scientifically
precise while keeping the CLI readable. `--symmetric` requests a restricted
square/hexagonal symmetry-preserving family and falls back to the general search
when inapplicable. Experimental N-layer commands are exposed separately as `moire stack-search`
and `moire stack-build`; their public contract is still being stabilized. See
the [moire search note](src/cellstine/moire/MOIRE_SEARCH.md) for the bilayer
algorithm, JSON fields, external Aristotle/Lean reference, and reproducibility
details.

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
- [Codebase inventory](docs/codebase-inventory.md)
- [Technical algorithms map](docs/technical/algorithms.md)
- [Moire search performance](docs/moire-performance.md)
- [Full usage guide](USAGE_GUIDE.md)
- [Roadmap](ROADMAP.md)

## Development

Run the test suite:

```bash
python -m pytest -q tests
```

Reproduce the three-bound canonical-class comparison and host-dependent timings:

```bash
python benchmarks/benchmark_gram_search.py
```

The benchmark compares independent canonical candidate classes, stops on
mismatch, and reports actual timings at three increasing length bounds.

The package uses a `src/` layout, with workflow code under `src/cellstine/` and regression tests under `tests/`.

## Author

Made by Sarwin Chandran, 2026.

## License

CELLSTINE is released under the [MIT License](LICENSE).
