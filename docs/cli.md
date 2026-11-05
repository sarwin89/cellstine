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

Answers are checked as you type them: paths must exist (quoted paths pasted from
a file manager are fine), numbers must be in range, and indexes, vectors, Miller
indices and matrices are parsed by the same rules as the direct command. Common
questions come first; rarely changed settings sit behind one "advanced options"
question. If a run fails or you press Ctrl+C, guided mode reports the error and
offers to reopen the command with every answer kept. Answering `n` at the run
confirmation returns to the menu. After a search, build, surface build or
interface match it offers the natural next step.

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

Bilayer search writes a portable NumPy archive by default:

```bash
cellstine moire search TOP.vasp BOTTOM.vasp --length 30 --strain 0.01
cellstine moire search TOP.vasp BOTTOM.vasp --length 30 --strain 0.01 --results-format json
cellstine moire search TOP.vasp BOTTOM.vasp --length 30 --strain 0.01 --results-format npz-compressed
```

The default file/artifact is `results.npz` / `results_npz`. Scripts that require
`results.json` / `results_json` must select `--results-format json`. Explicit
`--results-format npz` remains accepted. Compressed NPZ also writes
`results.npz`, trading encoding work for disk space. Molecular assembly exposes
the same option through `adsorbate assemble`.

`moire build` and `moire view` accept either format directly or through the run
manifest. Advanced guided search offers NPZ first and forwards an explicit JSON
choice. Python callers use `Moire.find(..., results_format="json")` or the same
keyword on `run_find` and `Adsorbate.assemble`; omitting it now selects NPZ.
These versioned archives preserve the validated JSON v2 result document and
never use pickle; CELLSTINE's `read_results` reconstructs that document. NumPy is
already a required dependency. Small results can remain faster as JSON; see the
[production measurements](optimization/part-05d-portable-results.md).

N-layer top-level results retain their separate experimental JSON schema;
intermediate bilayer pair results use NPZ. Interface v1 match records retain
actual JSON files for their existing `results_json` pointers. Existing JSON
inputs remain readable. No positional `.dat` compatibility is introduced.

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

Guided bilayer search saves every accepted candidate unless a shortlist is
requested. The direct Python `Moire.find` API retains its existing default of
500. The short guided rigid-search path retains the400-atom cap; advanced
questions allow an explicit opt-out. Back keeps earlier answers and reuses
previews; failed primary and follow-up commands can reopen those answers
without rerunning completed commands. Candidate index bounds use actual
validated saved rows, including empty-file handling.

Copied command previews use POSIX quoting on Unix hosts and literal PowerShell
quoting on Windows. PowerShell7 native argument roundtrips are checked. Legacy
PowerShell native quote handling and the reserved literal `--%` token remain
copy/paste limitations; guided execution passes arguments directly.

- `runs/<workflow>/<run-id>/manifest.json` records inputs, parameters, artifacts,
  backend, and summary metadata.
- `output/` stores generated structures, plots, site reports, and exported files.
- `input/` is for user source structures.
- `input/examples/` and `output/examples/` are tracked public examples.
