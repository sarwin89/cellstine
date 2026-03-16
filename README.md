# Moire Superstructure Toolkit

End-to-end commensurate moire workflow with:

- one **finder stage** (`moire/find.py`)
- one **maker stage** (`moire/make.py`)
- one user CLI wrapper (`moire_cli.py`)

Credits: **Made by Sarwin Chandran**.

## Design

The code is split into two core stages and shared common modules.

```text
.
+-- moire_cli.py              # single user-facing CLI (find + make)
+-- moire/
|   +-- find.py               # stage 1: find commensurate candidates
|   +-- make.py               # stage 2: generate superstructure POSCAR
|   +-- angles.py             # fast commensurate-angle shortlist utility
|   +-- finder.py             # vectorized candidate search backend
|   +-- generator.py          # exact structure construction backend
|   +-- lattice.py            # geometry, symmetry, strain helpers
|   +-- io.py                 # POSCAR read/write + coordinate transforms
|   +-- __init__.py
+-- tests/
+-- Reference/                # legacy code kept only for comparison
+-- Results/                  # reference outputs
```

## Features

- Commensurate angle search from integer spans up to `nindex`
- Strain-aware matching via relative vector-length mismatch tolerances
- Symmetry-aware angle range:
  - infer each lattice periodicity (`60`, `90`, or `180` degrees)
  - search in `[0, LCM(sym_top, sym_bottom)]`
- Candidate ranking and deduplication
- Exact POSCAR generation with user-selected candidate index
- Interactive prompts for bottom/top selection and interlayer spacing

## CLI Usage

The same script works on Windows, Linux, and macOS.

```bash
python moire_cli.py
python moire_cli.py find --help
python moire_cli.py make --help
```

Running `python moire_cli.py` opens the guided interactive workflow for normal use.
The wizard only requires the POSCAR paths and `nindex` up front, then offers recommended defaults for the rest.

### 1) Find commensurate candidates

```bash
python moire_cli.py find mos2.vasp mos2.vasp --nindex 12 --max-atoms 300
```

Optional explicit angles:

```bash
python moire_cli.py find mos2.vasp mos2.vasp --angles 13.15,21.787,27.9 --nindex 12
```

Important options:

- `--bottom a|b` choose which input goes below
- `--angle-strain-tolerance` shortlist tolerance on vector length mismatch
- `--vector-strain-tolerance` tolerance during vector-pair matching
- `--strain-tolerance` final strain filter on candidates
- `--output-root runs` where find artifacts are saved

If the fast exact-angle shortlist is empty, the finder automatically falls back to scanning the full symmetry-limited range `[0, LCM]`.

Find stage artifacts are saved into a timestamped run directory:

- `find_results.json`
- `find_results.md`
- `find_results.dat`

### 2) Make a final superstructure POSCAR

```bash
python moire_cli.py make runs/<run_name>/find_results.json --index 1 --interlayer 3.35
```

Output naming format:

- `stack_idx{index}_ang{angle}_atoms{count}_{bottom}-below_{top}-above.vasp`

## Notes on Strain Handling

- Angle shortlist can accept near-equal lengths through `--angle-strain-tolerance`
- Finder vector pairing uses `--vector-strain-tolerance`
- Final candidate filtering uses `--strain-tolerance` with `--strain-layer avg|1|2`

This gives practical strain tolerance in percentage-like relative mismatch terms.

## Testing

```bash
python -m unittest discover -s tests -q
```

Current tests validate:

- symmetry inference and LCM angle bounds
- MoS2 reference commensurate-angle families
- candidate atom counts near reference angles
- end-to-end `find -> make` workflow
- generated counts against saved `Results/` references
