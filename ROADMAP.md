# CELLSTINE Roadmap

This roadmap captures the next development wave after the structural cleanup pass. It prioritizes scientific workflow depth first, then reliability and publishability, then UX refinements.

## Phase 1: Scientific Workflow Expansion

### 1. Moire and Supermoire

- Consolidate the `find` and `findn` stages around clearer shared candidate validation while preserving the current CLI surface.
- Strengthen `base_shared`, `base_independent`, and `pairwise` validation so buildable `N`-layer candidates are previewed more explicitly before generation.
- Expand per-layer prestrain handling with clearer validation, reporting, and candidate filtering.
- Improve matrix-based filtering and reporting so users can reason about candidate matrices without post-processing.
- Tighten multilayer translation and reframing logic for large or low-symmetry stacks.

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

- Add stronger regression fixtures that check output logic, not just command success.
- Add packaging and release automation for install, build, and metadata verification.
- Expand native-vs-optional-backend parity checks where `pymatgen` or `spglib` overlaps native workflows.
- Tighten public API boundaries for future notebooks, scripting, and plugin integrations.

## Phase 3: UX and Visualization

- Improve manifest-to-next-step chaining inside the guided CLI.
- Add clearer candidate previews before generation for moire, interface, adsorbate, and defect workflows.
- Expand Matplotlib-first scientific diagnostics and publication-friendly summary plots.
- Keep Plotly optional and focused on cases where interactive 3D inspection materially adds value.
