# CELLSTINE Usage Guide

This guide covers the current grouped CELLSTINE package and CLI in more detail than the README.
For a shorter public documentation map, see [docs/README.md](docs/README.md).

The package is organized around six top-level workflows:

- `moire`
- `adsorbate`
- `surface`
- `interface`
- `symmetry`
- `defect`

Examples below use the installed command `cellstine`. Public example files live in `input/examples/` and `output/examples/`; normal local inputs, generated outputs, and run manifests stay in `input/`, `output/`, and `runs/`. Inside the repository, `python cellstine.py ...` is also available as a repository-local convenience entrypoint that forwards into the same package CLI.

Import workflow classes and helpers from `cellstine...` modules.

## 1. Installation And Entry Points

Base install:

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[pymatgen]"
pip install -e ".[symmetry]"
pip install -e ".[viz]"
pip install -e ".[plotly]"
pip install -e ".[cli]"
pip install -e ".[all]"
```

Main entry points:

```bash
cellstine --help
cellstine --version
python cellstine.py --help
```

## 2. Standard Working Layout

CELLSTINE assumes this folder structure by default:

```text
input/   -> source structures and local sample POSCARs
runs/    -> manifests and saved workflow artifacts
output/  -> generated POSCARs, slabs, interfaces, moved structures, static plots, and optional HTML viewers
```

Each grouped stage writes a manifest to:

```text
runs/<workflow>/<run-id>/manifest.json
```

Build stages can consume either a raw results file or a manifest from the previous stage.
Generated outputs include a run-specific identifier with a short `yymmdd-hhmm` timestamp at the end of the filename, or are written into a run-specific output subfolder, which keeps repeated tests with different parameters from overwriting each other.

## 3. Interactive Interface

Launch the guided interface with:

```bash
cellstine
```

You can also jump directly into one workflow group and let CELLSTINE ask only for that group:

```bash
cellstine moire
cellstine adsorbate
cellstine interface
cellstine symmetry
cellstine defect
```

The interactive flow is now grouped:

1. choose `moire`, `adsorbate`, `interface`, `symmetry`, or `defect`
2. choose the stage within that workflow
3. enter only the inputs needed for that stage
4. use the manifest or generated artifact for the next step

The guided picker uses the standard folder flow:

- new source structures are suggested from `input/` first
- generated slabs, interfaces, adsorbates, and stacked structures are suggested from `output/` first
- saved searches, manifests, and intermediate workflow artifacts are suggested from `runs/` first

At any picker, type `b` to return to the previous menu, or type `q`, `quit`, or `exit` to close the interactive interface cleanly.

It is meant to be a guided launcher over the same backend classes used by the CLI, not a separate hidden workflow.

## 4. CLI Help Pages

```bash
cellstine --help
cellstine moire --help
cellstine moire search --help
cellstine moire build --help
cellstine moire shift --help
cellstine moire view --help
cellstine moire stack-search --help
cellstine moire stack-build --help
cellstine adsorbate --help
cellstine adsorbate place --help
cellstine adsorbate move --help
cellstine adsorbate assemble --help
cellstine surface --help
cellstine surface build --help
cellstine surface sites --help
cellstine interface --help
cellstine interface build --help
cellstine interface match --help
cellstine interface registries --help
cellstine symmetry --help
cellstine symmetry analyse --help
cellstine symmetry reduce --help
cellstine symmetry lattice-reduce --help
cellstine symmetry kpoints --help
cellstine symmetry kpath --help
cellstine defect --help
cellstine defect analyse --help
cellstine defect generate --help
cellstine defect preview --help
cellstine view --help
```

## 5. `moire` Workflow

### 5.1 Native Bilayer Search

Run `moire search` with physical length and layer strain budgets:

```bash
cellstine moire search input/top.vasp input/bottom.vasp --length 20 --strain 0.01
```

Useful optional controls include `--min-length`, `--atoms`, `--twist`,
`--max-cell-aspect-ratio`, `--min-cell-angle`, `--max-cell-angle`,
`--symmetric`, `--progress`, and `--preview-limit`. Twist angles are
filtered in degrees with ranges such as `--twist 9:14`. Use `--rigid` for zero
strain, `--strain E` for equal layer budgets, or `--top-strain` with
`--bottom-strain` for asymmetric expert searches. Lengths are in angstrom and
strain budgets are fractions, so `0.01` means a
logarithmic-strain budget of 0.01.

Here **strain** means the principal logarithmic strain
`h = log(lambda)` of the relative deformation's principal stretch `lambda`.
The accepted relative principal strain is bounded by the sum of the top and
bottom strain budgets, and the engine shares it optimally between the layers for
each candidate. This naming is deliberate: it is scientifically precise while
keeping the CLI readable; it does not mean engineering strain.

`--symmetric` requests a restricted square/hexagonal symmetry-preserving
family. When that family is inapplicable, CELLSTINE records why and falls back
to the general search.

### 5.2 JSON Results And `moire build`

Every successful search writes schema-versioned
`cellstine.moire.gram` JSON v1:

```text
runs/moire/<run-id>/results.json
runs/moire/<run-id>/manifest.json
```

The JSON carries the candidate matrices, angle in degrees, relative principal
strain, top and bottom strain budgets, atom counts, rank/Pareto status, Löwner
certification, shared lattice, affine maps, and search metadata. It is the
single handoff format for previews, visualization, and construction. Legacy
positional `.dat` files are rejected; rerun native `moire search` to create
`results.json`.

Build one or several selected candidates:

```bash
cellstine moire build runs/moire/<run-id>/results.json --indexes 1 --interlayer-distance 3.35
cellstine moire build runs/moire/<run-id>/results.json --indexes 1,2,5-7 --interlayer-distance 3.35 --workers 4
```

A manifest containing the `results_json` artifact can be passed instead of
the raw JSON path.

### 5.3 Translation, Preview, And Visualization

Shift the upper layer of an existing bilayer:

```bash
cellstine moire shift output/stacked.vasp --shift-direct 0.333,0.667
```

Create a labelled static summary or the optional interactive Plotly viewer:

```bash
cellstine moire view runs/moire/<run-id>/results.json --indices 1,2,3
cellstine moire view runs/moire/<run-id>/results.json --indices 1,2,3 --plotly
```

Both paths use the validated JSON reader. They label top strain, bottom strain,
and relative principal strain explicitly and expose candidate provenance rather
than interpreting positional columns. The guided interface is a simple launcher
over these same commands and data.

Experimental N-layer workflows are exposed as `moire stack-search` and
`moire stack-build`. Treat them as contract-in-progress until their native JSON
schema, oracle tests, and construction guarantees are documented to the same
standard as the bilayer `moire search` and JSON `moire build` sequence.

### 5.4 Mathematical Reference And Benchmark

The native implementation was informed by external mathematical reference work
with [Harmonic's Aristotle](https://aristotle.harmonic.fun/) and
[Lean 4](https://lean-lang.org/lean4/doc/). Those tools are not runtime
dependencies, and no Lean source files are copied into this repository.

Reproduce the comparison from the repository root:

```bash
python benchmarks/benchmark_gram_search.py
```

The benchmark compares independent canonical candidate classes, stops on
mismatch, reports actual timings for three increasing length bounds, and notes
that speed numbers are host-dependent.


## 6. `adsorbate` Workflow

### 6.1 Placement With `adsorbate place`

The substrate input can be:

- `substrate`
- `patch`
- `surface`
- `slab`
- `bulk`

If you pass `bulk`, CELLSTINE first generates a surface slab through the `surface build` machinery.
For bulk-derived substrates, the primitive surface is generated first; then optional `--substrate-repeat-a`, `--substrate-repeat-b`, or `--substrate-supercell-matrix` expansion is applied before site detection and placement.

Place on an existing slab:

```bash
cellstine adsorbate place output/surface_Au_Bulk_111_layers4.vasp input/papd_gasp_mol2_final-coor_at_.vasp --site-type fcc --site-index 1 --height 2.3
```

Place on a bulk-derived substrate:

```bash
cellstine adsorbate place input/Au_Bulk.vasp input/papd_gasp_mol2_final-coor_at_.vasp --substrate-kind bulk --miller 1,1,1 --layers 4 --vacuum 15 --site-type top --height 2.0
```

Place on a bulk-derived substrate with a non-diagonal in-plane surface matrix:

```bash
cellstine adsorbate place input/Au_Bulk.vasp input/molecule.vasp --substrate-kind bulk --miller 111 --layers 4 --substrate-supercell-matrix 1,1,0,2 --site-type top --height 2.5
```

Place on a primitive slab and let CELLSTINE enlarge the substrate patch if the molecule is too wide for one periodic image:

```bash
cellstine adsorbate place output/Au_111_primitive.vasp input/molecule.vasp --site-type top --height 2.5 --auto-repeat-substrate
```

How placement works:

- the site list is determined from the slab geometry
- guided mode analyses the chosen substrate first and only offers site families found in that specific cell
- the molecule is rotated rigidly about its center of mass
- the molecule is aligned in-plane to the chosen site
- the closest molecule atom is then placed `--height` angstrom above the selected surface plane
- without `--auto-repeat-substrate`, CELLSTINE rejects molecules that cannot fit into one periodic image of the substrate cell instead of folding them into many apparent copies

This means `--height` is a molecule-to-surface gap, not a COM height.

### 6.2 Rigid Molecular Movement With `adsorbate move`

Move a top-side molecule by Direct coordinates:

```bash
cellstine adsorbate move output/stacked.vasp --target-direct 0.5,0.5 --rotate 30
```

Move by Cartesian coordinates:

```bash
cellstine adsorbate move output/stacked.vasp --target-cart 12.0,8.0,10.5 --rotate 45
```

### 6.3 Molecular Assembly With `adsorbate assemble`

This mode uses an experimental target lattice to search for a commensurate substrate supercell beneath it.

```bash
cellstine adsorbate assemble input/Au_1x1.vasp --a-length 12.0 --b-length 12.0 --angle 60 --length 30 --top-strain 0.05 --bottom-strain 0.05
```

`--top-strain` bounds the synthetic molecular lattice and `--bottom-strain`
bounds the substrate, using the same principal logarithmic strain definition as
native bilayer moire search. The command writes schema-versioned `results.json`.

### 6.4 Adsorbate Visualization

```bash
cellstine view output/stacked.vasp
```

This writes a labelled Matplotlib multi-view PNG by default: top view `x-y`, side view `x-z`, side view `y-z`, and a 3D overview. Add `--plotly` only when you want an optional interactive 3D HTML viewer.

## 7. `interface` Workflow

### 7.1 Surface Generation With `surface build`

Build an `Au(111)`-style slab:

```bash
cellstine surface build input/Au_Bulk.vasp --miller 111 --layers 6 --vacuum 15
```

CELLSTINE first builds the primitive surface cell for the requested direction, then applies any requested repeats or supercell matrix. For conventional fcc metals such as Au, Au(111) with four layers gives one atom per layer and `ABCA` stacking; Au(001) with four layers gives `ABAB`. Use `--repeat-a`, `--repeat-b`, or `--supercell-matrix` when you intentionally want a larger surface patch.

In guided interactive mode, the surface workflow previews the primitive cell first: detected centering, stacking sequence, repeating stacking period, atoms per layer, and in-plane angle. It then asks whether to keep the primitive cell, repeat it, or apply a matrix, followed by layer count and vacuum.

`--vacuum` is the empty space above the slab. CELLSTINE places the bottom layer one interlayer spacing above the lower cell boundary, so the same spacing exists below the slab while the requested vacuum remains above the active surface.

Miller notation can be compact or comma-separated:

```bash
cellstine surface build input/Au_Bulk.vasp --miller 001 --layers 4
cellstine surface build input/Au_Bulk.vasp --miller 111x --layers 4
cellstine surface build input/Au_Bulk.vasp --miller 1,1,2x --layers 6
```

Repeat in plane:

```bash
cellstine surface build input/Au_Bulk.vasp --miller 110 --layers 4 --repeat-a 2 --repeat-b 2
```

Apply an explicit in-plane `2x2` supercell matrix:

```bash
cellstine surface build input/Au_Bulk.vasp --miller 100 --layers 6 --supercell-matrix 2,0,0,3
```

Also write the adsorption-site report:

```bash
cellstine surface build input/Au_Bulk.vasp --miller 111 --layers 6 --vacuum 15 --analyse-sites
```

Native surface generation detects primitive, body-centred, face-centred, and base-centred translational lattices from the input structure before cutting the surface slab.

### 7.2 Site Analysis With `surface sites`

Analyse an existing slab:

```bash
cellstine surface sites output/surface_Au_Bulk_111_layers4.vasp
cellstine surface sites output/surface_Au_Bulk_111_layers4.vasp --surface-side bottom
```

The site finder reports:

- `top`
- `bridge`
- `hcp_hollow`
- `fcc_hollow`
- `hollow`
- `fourfold_hollow`

For close-packed surfaces such as `(111)`, the native classifier distinguishes `hcp` and `fcc` hollows by subsurface registry.

### 7.3 Interface Construction With `interface build`

Build from two existing slabs:

```bash
cellstine interface build output/bottom_slab.vasp output/top_slab.vasp --gap 3.0
```

Build from two bulks:

```bash
cellstine interface build input/Au_Bulk.vasp input/Au_Bulk.vasp --bottom-kind bulk --top-kind bulk --bottom-miller 1,1,1 --top-miller 1,0,0 --bottom-layers 6 --top-layers 4 --gap 3.0
```

Behavior:

- the bottom slab is fixed
- the top slab is strained in-plane to the bottom slab lattice
- the final heterostructure is written as a VASP structure in `output/`

### 7.4 Bulk Surface Matching With `interface match`

Scan surface combinations from two bulks:

```bash
cellstine interface match input/Au_Bulk.vasp input/Au_Bulk.vasp --bottom-millers 1,1,1 1,0,0 --top-millers 1,1,1 1,1,0 --bottom-layers-list 4 6 --top-layers-list 4 6 --max-strain 0.05
```

Ranking priority:

1. strain
2. total atom count
3. surface area

This means smaller commensurate cells are preferred when strain is otherwise comparable.

### 7.5 Interface Visualization

```bash
cellstine view output/interface.vasp
```

This uses the same labelled Matplotlib multi-view format as adsorbate visualization, which is usually more useful for checking slab orientation, vacuum, adsorption height, and layer separation than a free-rotating viewer alone.

## 8. `symmetry` Workflow

The `symmetry` workflow keeps symmetry operations out of the older `pymatgen` path. It uses direct `spglib` when installed, and falls back to a native metadata report for environments where exact symmetry detection is unavailable.

### 8.1 Analyse Symmetry

```bash
cellstine symmetry analyse input/Au_Bulk.vasp --backend auto --symprec 0.01
```

The report includes lattice parameters, species counts, space group symbol and number, Hall symbol, point group, crystal system, rotations, translations, Wyckoff letters, equivalent atom groups, transformation matrix, origin shift, and backend notes when available.

Use `--backend native` when you only want a dependency-free structure summary with a clear note that exact symmetry is unavailable.

### 8.2 Reduce Cells

Primitive reduction:

```bash
cellstine symmetry reduce input/Au_Bulk.vasp --cell primitive --output output/Au_primitive.vasp
```

Standard conventional cell:

```bash
cellstine symmetry reduce input/Au_Bulk.vasp --cell conventional --output output/Au_conventional.vasp
```

Refined cell:

```bash
cellstine symmetry reduce input/Au_Bulk.vasp --cell refined --output output/Au_refined.vasp
```

Reduction requires `spglib`; install it with `pip install -e ".[symmetry]"` or `pip install -e ".[all]"`.

### 8.3 Lattice Reduction

```bash
cellstine symmetry lattice-reduce input/Au_Bulk.vasp --reduction niggli --output output/Au_niggli.vasp
cellstine symmetry lattice-reduce input/Au_Bulk.vasp --reduction delaunay --output output/Au_delaunay.vasp
```

These commands are useful for cleaning lattice bases before comparing structures or preparing follow-up symmetry analysis.

## 9. `defect` Workflow

The `defect` workflow analyses valid defect sites first, then generates structures from the discovered site IDs. This is deliberately a two-step flow so users are not asked to guess site indices blindly.

### 9.1 Analyse Defect Sites

```bash
cellstine defect analyse output/Au_Bulk_111_surface.vasp --structure-kind surface
```

Useful options:

- `--structure-kind auto|bulk|surface|slab|molecule-on-substrate`
- `--backend auto|native|spglib`
- `--surface-side top|bottom`
- `--layer-tolerance 0.35`
- `--symprec 0.01`

Backend behavior:

- `auto` uses direct `spglib` for bulk equivalence when it is installed.
- slab and surface inputs prefer native layer-aware grouping because 3D bulk symmetry is often the wrong mental model for a finite slab.
- exact Wyckoff labels are only guaranteed with the `spglib` backend.

The analysis writes:

```text
runs/defect/<run-id>/manifest.json
runs/defect/<run-id>/defect_analysis.json
```

### 9.2 Preview Site IDs

```bash
cellstine defect preview runs/defect/<run-id>/manifest.json
```

The preview table lists:

- `site_id`
- defect-site kind: `atom`, `interstitial`, or `adatom`
- species and layer where relevant
- multiplicity and represented atom indices
- Wyckoff label when available from `spglib`

### 9.3 Generate Vacancy And Substitution Defects

Generate one vacancy POSCAR from one inequivalent atom site:

```bash
cellstine defect generate runs/defect/<run-id>/manifest.json --defect-type vacancy --site-ids atom_001
```

Generate all inequivalent vacancies:

```bash
cellstine defect generate runs/defect/<run-id>/manifest.json --defect-type vacancy
```

Generate a substitution:

```bash
cellstine defect generate runs/defect/<run-id>/manifest.json --defect-type substitution --site-ids atom_001 --substitution-species Pt
```

Restrict substitutions to one original species:

```bash
cellstine defect generate runs/defect/<run-id>/manifest.json --defect-type substitution --original-species Au --substitution-species Pt
```

### 9.4 Generate Interstitials And Adatoms

Generate interstitial candidate structures:

```bash
cellstine defect generate runs/defect/<run-id>/manifest.json --defect-type interstitial --species H
```

Generate an adatom on a detected surface site:

```bash
cellstine defect generate runs/defect/<run-id>/manifest.json --defect-type adatom --site-ids adatom_fcc_hollow_001 --species H --height 2.2
```

For adatoms, `--height` is in angstrom above the detected top/bridge/hollow site. If the chosen height falls outside the current cell, increase the slab vacuum first.

Generated structures are written to:

```text
output/defect_<type>_<site-id>_<species-info>_<yymmdd-hhmm>.vasp
```

## 10. Python API Examples

The grouped classes are also usable directly from Python.

Bilayer moire:

```python
from cellstine import Moire

result = Moire().find(
    top_poscar="input/mos2.vasp",
    bottom_poscar="input/mos2.vasp",
    max_length=20.0,
    top_strain=0.01,
    bottom_strain=0.01,
)
print(result.manifest_path)
print(result.artifacts["results_json"])
```

Adsorbate placement:

```python
from cellstine import Molecule

result = Molecule().place(
    substrate_poscar="input/Au_Bulk.vasp",
    molecule_poscar="input/papd_gasp_mol2_final-coor_at_.vasp",
    substrate_kind="bulk",
    miller="1,1,1",
    layers=4,
    site_type="fcc",
    height=2.3,
)
print(result.artifacts["output_poscar"])
```

Surface generation:

```python
from cellstine import Surface

result = Surface().surface(
    bulk_poscar="input/Au_Bulk.vasp",
    miller="1,1,1",
    layers=6,
    vacuum=15.0,
    analyse_sites=True,
)
print(result.summary)
```

Defect analysis and generation:

```python
from cellstine import Defect

analysis = Defect().analyse(
    structure_path="output/Au_Bulk_111_surface.vasp",
    structure_kind="surface",
)

generated = Defect().generate(
    str(analysis.manifest_path),
    defect_type="vacancy",
    site_ids=["atom_001"],
)
print(generated.artifacts["structures"])
```

Symmetry analysis and reduction:

```python
from cellstine import Symmetry

symmetry = Symmetry()
analysis = symmetry.analyse("input/Au_Bulk.vasp", backend="auto")
print(analysis.summary["space_group"])

reduced = symmetry.reduce(
    "input/Au_Bulk.vasp",
    cell="primitive",
    output_path="output/Au_primitive.vasp",
)
print(reduced.artifacts["output_poscar"])
```

## 11. Optional Dependencies And Backends

Current backend behavior:

- `numpy` is required
- VASP I/O is native and always available
- XYZ conversion is handled natively
- `spglib` is used directly for exact symmetry analysis, cell reduction, Wyckoff labels, and bulk defect equivalence when installed
- `pymatgen` is temporarily used only for broad non-VASP structure conversion when installed
- `matplotlib` is used by default for static, labelled PNG plots
- `plotly` is secondary and only used when `--plotly` is requested for interactive 3D HTML viewers

Check installed versions:

```bash
cellstine --version
```

## 12. Testing

Run the current test suite with:

```bash
python -m pytest -q tests
```

## 13. Troubleshooting

If a moire search returns too many candidates:

- lower the top or bottom strain budget
- lower `--length`
- lower `--atoms`
- tighten the cell-shape limits

If slab generation fails:

- check that the bulk input is conventional and orthogonal
- start with simple Miller planes like `1,0,0`, `1,1,0`, or `1,1,1`

If a visualization command says Matplotlib is missing:

- install the visualization extra with `pip install -e ".[viz]"`
- or rerun with `--plotly` if you want the optional HTML viewer instead

If a Plotly visualization HTML opens blank:

- verify the results file actually contains candidates
- try a single index first
- if you are offline, remember that the optional HTML viewer uses the Plotly CDN
