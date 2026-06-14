# Quickstart

## Install

From the repository root:

```bash
python -m pip install -e .
```

Install optional scientific and visualization backends when needed:

```bash
python -m pip install -e ".[all]"
```

## Check The CLI

```bash
cellstine --help
cellstine --version
```

Running `cellstine` without arguments starts the guided interface.

## First Commands

Find commensurate MoS2/MoS2 cells:

```bash
cellstine moire find input/examples/mos2.vasp input/examples/mos2.vasp --nindex 8
```

Generate an Au(111) slab:

```bash
cellstine interface surface input/examples/Au_Bulk.vasp --miller 111 --layers 4 --vacuum 15
```

Inspect adsorption sites on the curated Au(111) example:

```bash
cellstine interface sites output/examples/Au_Bulk_111_surface.vasp
```

Analyse symmetry:

```bash
cellstine symmetry analyse input/examples/Au_Bulk.vasp
```

## Folder Layout

```text
input/examples/    Public sample inputs
output/examples/   Public sample outputs
input/             Local user inputs
output/            Generated structures and plots
runs/              Manifest folders and intermediate artifacts
```
