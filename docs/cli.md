# CLI And Interactive Mode

CELLSTINE has both a guided interface and direct subcommands.

## Guided Mode

```bash
cellstine
```

The guided interface asks for one workflow first, then asks only for inputs relevant to that stage. It prefers source structures from `input/`, generated structures from `output/`, and manifests from `runs/`. Use `b` in menus that support back-navigation.

## Direct Commands

```bash
cellstine moire --help
cellstine adsorbate --help
cellstine interface --help
cellstine defect --help
cellstine symmetry --help
```

Every subcommand has `-h/--help` text with units, defaults, and examples.

## Output Model

- `runs/<workflow>/<run-id>/manifest.json` records inputs, parameters, artifacts, backend, and summary metadata.
- `output/` stores generated structures, plots, site reports, and exported files.
- `input/` is for user source structures.
- `input/examples/` and `output/examples/` are tracked public examples.
