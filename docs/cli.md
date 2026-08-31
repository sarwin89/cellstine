# CLI and Interactive Mode

CELLSTINE has one command surface rendered by two frontends:

- base install: dependency-free plain CLI;
- `pip install -e ".[cli]"`: Rich/Typer presentation for guided use, clearer
  help, and styled summaries.

Use `--plain` to force the stdlib frontend.

The installed public command is only:

```bash
cellstine
```

When running directly from a repository checkout, use the local launcher:

```bash
python cellstine.py
```

There is no separate moiré-only command; moiré workflows live under
`cellstine moire ...`.

## Guided mode

```bash
cellstine
```

Guided mode asks what you are trying to do, infers conservative defaults, prints
the command it generated, and asks before running it. It does not mutate files
or launch a workflow until you confirm.

## Direct commands

```text
cellstine moire search
cellstine moire build
cellstine moire shift
cellstine moire view
cellstine moire stack-search
cellstine moire stack-build
cellstine surface build
cellstine surface sites
cellstine interface match
cellstine interface build
cellstine interface registries
cellstine adsorbate place|move|path|assemble
cellstine defect analyse|generate|supercell|preview|path
cellstine symmetry analyse|reduce|kpoints|kpath
cellstine view STRUCTURE
```

Moiré search uses the readable flags:

```bash
cellstine moire search TOP.vasp BOTTOM.vasp --length 20 --strain 0.01
cellstine moire search TOP.vasp BOTTOM.vasp --length 30 --rigid --twist 9:14 --atoms 400
```

Choose exactly one strain mode: `--rigid`, `--strain E`, or both
`--top-strain` and `--bottom-strain`.

## Migration table

| Old command | New command |
| --- | --- |
| `cellstine moire find` | `cellstine moire search` |
| `cellstine moire make` | `cellstine moire build` |
| `cellstine moire translate` | `cellstine moire shift` |
| `cellstine moire visualize` | `cellstine moire view` |
| `cellstine moire findn` | `cellstine moire stack-search` |
| `cellstine moire maken` | `cellstine moire stack-build` |
| `cellstine interface surface` | `cellstine surface build` |
| `cellstine interface sites` | `cellstine surface sites` |
| `<workflow> visualize STRUCTURE` | `cellstine view STRUCTURE` |

For one release, removed commands fail early with this replacement guidance
instead of falling through to ambiguous parser errors.

## Output model

- `runs/<workflow>/<run-id>/manifest.json` records inputs, parameters, artifacts,
  backend, and summary metadata.
- `output/` stores generated structures, plots, site reports, and exported files.
- `input/` is for user source structures.
- `input/examples/` and `output/examples/` are tracked public examples.
