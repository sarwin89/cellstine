# CELLSTINE

<p align="center">
  <!--
  Logo placeholder:
  Add your logo at docs/logo.png and uncomment the image below.
  <img src="docs/logo.png" alt="CELLSTINE logo" width="180">
  -->
</p>

<p align="center">
  <strong>CELL Superlattice Transformation INterface and Engine</strong>
</p>

<p align="center">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Version 4.0.0" src="https://img.shields.io/badge/version-4.0.0-informational">
</p>

CELLSTINE is a Python package and guided command-line tool for building atomistic structures used in VASP-style materials workflows. It provides grouped workflows for commensurate moire supercells, molecule-on-substrate placement, surface generation, adsorption-site analysis, slab-on-slab interface construction, and defect-structure generation.

The codebase now lives fully under `src/cellstine`, with the earlier top-level `moire/` compatibility layer retired after migration into the package modules.

## Features

- Commensurate bilayer and N-layer moire search and construction.
- Adsorbate placement, movement, rotation, and substrate patch handling.
- Bulk-to-surface slab generation and adsorption-site detection.
- Heterointerface construction and surface-match screening.
- Inequivalent defect-site analysis and generation of vacancy, substitution, interstitial, antisite, and adatom structures.
- Native VASP-style I/O with optional `pymatgen`, `matplotlib`, and `plotly` support.
- Manifest-based runs for reproducible workflow chaining.

## Installation

Install from the repository:

```bash
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[pymatgen]"
python -m pip install -e ".[viz]"
python -m pip install -e ".[plotly]"
python -m pip install -e ".[all]"
```

Check the installation:

```bash
cellstine --help
cellstine --version
```

## Usage

Start the guided interface:

```bash
cellstine
```

Or jump directly to one workflow group:

```bash
cellstine moire
cellstine adsorbate
cellstine interface
cellstine defect
```

Every command also has a help page:

```bash
cellstine moire --help
cellstine adsorbate --help
cellstine interface --help
cellstine defect --help
```

Detailed CLI examples and workflow notes are in [USAGE_GUIDE.md](USAGE_GUIDE.md).

## Repository Layout

```text
src/cellstine/   Main Python package and workflow backends
tests/           Unit tests
input/           Local source structures
runs/            Saved manifests and intermediate workflow artifacts
output/          Generated structures, plots, viewers, and reports
```

## Development

Run the test suite:

```bash
python -m unittest discover -s tests -q
```

Generated structures, plots, run folders, and local notebooks are ignored by default. Keep committed examples small and intentional.

## Citation

If CELLSTINE supports published work, cite the repository or the relevant release until a formal paper or DOI is available.

## Author

Sarwin Chandran

## License

CELLSTINE is released under the [MIT License](LICENSE).
