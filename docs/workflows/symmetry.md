# Symmetry Workflow

The `symmetry` workflow analyses equivalent atom groups and writes reduced cells.

## Common Commands

```bash
cellstine symmetry analyse input/examples/Au_Bulk.vasp
cellstine symmetry reduce input/examples/Au_Bulk.vasp --cell primitive --output output/Au_primitive.vasp
cellstine symmetry lattice-reduce input/examples/Au_Bulk.vasp --reduction niggli --output output/Au_niggli.vasp
```

## Optional Backend

Install `spglib` support with:

```bash
python -m pip install -e ".[symmetry]"
```

CELLSTINE keeps native fallbacks where possible, but richer space-group metadata depends on optional symmetry libraries.
