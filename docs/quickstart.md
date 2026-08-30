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

For the recommended Rich/Typer interface without every scientific extra:

```bash
python -m pip install -e ".[cli]"
```

## Check the CLI

```bash
cellstine --help
cellstine --version
```

Running `cellstine` without arguments starts the guided interface.

## First commands

Find native Gram-form MoS2/MoS2 bilayer candidates, then build candidate 1
from the schema-versioned JSON:

```bash
cellstine moire search input/examples/mos2.vasp input/examples/mos2.vasp --length 20 --strain 0.01
cellstine moire build runs/moire/<run-id>/results.json --indexes 1 --interlayer-distance 3.35
```

Here moire **strain** is principal logarithmic strain, `h = log(lambda)`, for
principal stretch `lambda`. The accepted relative strain is bounded by the sum
of the two layer budgets and shared optimally between the layers. `--symmetric`
requests the restricted square/hexagonal family and falls back to the general
search when it is inapplicable. N-layer moire workflows are not supported in
this release. See the [moire workflow](workflows/moire.md) for details and the
[performance note](moire-performance.md) for the reproducible benchmark.

Generate an Au(111) slab:

```bash
cellstine surface build input/examples/Au_Bulk.vasp --miller 111 --layers 4 --vacuum 15
```

Inspect adsorption sites on the curated Au(111) example:

```bash
cellstine surface sites output/examples/Au_Bulk_111_surface.vasp
```

Analyse symmetry:

```bash
cellstine symmetry analyse input/examples/Au_Bulk.vasp
```

## Folder layout

```text
input/examples/    Public sample inputs
output/examples/   Public sample outputs
input/             Local user inputs
output/            Generated structures and plots
runs/              Manifest folders and intermediate artifacts
```
