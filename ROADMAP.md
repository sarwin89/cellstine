# CELLSTINE Roadmap

This roadmap captures the next development wave after the structural cleanup
pass. It prioritizes public contracts and scientific correctness first, then
reliability and publishability, then UX refinements. For the current repository
shape, see [docs/codebase-inventory.md](docs/codebase-inventory.md); for the
algorithm map, see [docs/technical/algorithms.md](docs/technical/algorithms.md).

## Phase 0: Contract And Organization Cleanup

- Keep `src/cellstine/` as the source package and avoid another large donor
  folder inside `src/`.
- Keep local proof/reference and benchmark corpora untracked; cite them in docs
  only where they affect public algorithms or validation.
- Decide whether experimental N-layer stack commands are kept public in the
  next release or hidden until their JSON contract and construction guarantees
  match the bilayer workflow.
- Split oversized modules only along stable domain boundaries: parsing, lattice
  algebra, search kernels, workflow orchestration, and presentation.
- Keep command names and public docs aligned with the simplified CLI surface:
  `moire search/build/shift/view`, `surface build/sites`,
  `interface match/build/registries`, `symmetry analyse/reduce/kpoints/kpath`,
  and the root `view` command.

## Phase 1: Scientific Workflow Expansion

### 1. Moire

- Deepen validation and reporting for the native bilayer Gram-form
  `moire search` to JSON `moire build` workflow.
- Improve matrix-based filtering and reporting so users can reason about candidate matrices without post-processing.
- Stabilize or hide the experimental `moire stack-search` and
  `moire stack-build` commands. If stabilized, define a native versioned JSON
  schema, independent oracle tests, affine construction checks, and docs before
  treating them as first-class public workflows.

### 2. Adsorbate and Molecule-on-Substrate

- Improve automatic substrate patch sizing so molecules are expanded onto the smallest sensible repeated slab.
- Surface only the site families that actually exist in the analysed slab before prompting for placement.
- Surface arbitrary-axis rigid rotation more clearly in the guided flow.
- Improve input-kind handling for bulk, slab, primitive surface, and full substrate patch cases.
- Add stronger validation around COM-driven movement and post-rotation reframing.

### 3. Interface and Surface

- Expand primitive-surface generation coverage across more lattice types and centering cases.
- Improve stacking-sequence identification and reporting for reconstructed or lower-symmetry surfaces.
- Strengthen interface matching and ranking across bulk-derived surface combinations.
- Improve equivalence reporting for adsorption site families on generated slabs.
- Tighten interface build rules around in-plane strain, layer counts, gap handling, and vacuum conventions.

### 4. Defect

- Improve transparency of native equivalence grouping, especially for lower-symmetry bulk structures.
- Add richer defect families such as antisites, paired vacancies, and constrained-layer variants.
- Expand manifests and site metadata so generated defect sets are easier to trace and reuse.
- Improve slab-aware defect generation for adatoms, surface interstitials, and layer-restricted workflows.

## Phase 2: Reliability and Publishability

- Add a small documentation/link-check gate so public examples do not drift
  from the CLI spec again.
- Add stronger regression fixtures that check output logic, not just command success.
- Add packaging and release automation for install, build, and metadata verification.
- Expand native-vs-optional-backend parity checks where `pymatgen` or `spglib` overlaps native workflows.
- Tighten public API boundaries for future notebooks, scripting, and plugin integrations.

## Phase 3: UX and Visualization

- Improve manifest-to-next-step chaining inside the guided CLI.
- Add clearer candidate previews before generation for moire, interface, adsorbate, and defect workflows.
- Expand Matplotlib-first scientific diagnostics and publication-friendly summary plots.
- Keep Plotly optional and focused on cases where interactive 3D inspection materially adds value.
