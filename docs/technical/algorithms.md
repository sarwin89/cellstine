# Technical algorithms map

This is the canonical map of CELLSTINE's mathematical and algorithmic
implementation. It does not replace the detailed source-adjacent notes; it links
them together and names the code that implements each idea.

Conventions used throughout:

- Atomic structures are represented with row-vector lattice conventions in the
  Python package.
- Cartesian coordinates are obtained from Direct/fractional coordinates by the
  package's structure models and I/O helpers.
- In-plane moire and interface comparisons use principal logarithmic strain:
  `h = log(lambda)`, where `lambda` is a principal stretch.
- Generated workflow state is recorded through manifests and schema-versioned
  JSON result files rather than positional text tables.

## 1. Core periodic geometry

Main code:

- `src/cellstine/core/geometry.py`
- `src/cellstine/core/reduction.py`
- `src/cellstine/core/transforms.py`
- `src/cellstine/core/lattice.py`
- `src/cellstine/core/contacts.py`

Detailed notes:

- `src/cellstine/core/GEOMETRY.md`
- `src/cellstine/core/CONTACTS.md`

Key functions and objects:

| Code | Algorithmic role |
| --- | --- |
| `shortest_lattice_vector_length`, `shortest_plane_vector_length` | Bound the shortest periodic vectors used in search radii and validation. |
| `image_shift_reach`, `plane_shift_reach` | Compute a complete integer-image search box for periodic distances. |
| `lattice_shifts`, `plane_shifts` | Enumerate image shifts inside those complete bounds. |
| `plane_minimum_image`, `minimum_image_displacements`, `minimum_image_distances` | Exact periodic minimum-image calculations; do not rely on naive rounding. |
| `bounded_minimum_image_squared` | Fast bounded distance check for pruning. |
| `periodic_midpoints`, `atom_images` | Generate geometrically meaningful image-aware sites. |
| `plane_form_kernel_basis`, `gauss_reduction_multiplier`, `plane_reduce` | Integer lattice and 2D reduction utilities used by surfaces, moire, and defects. |
| `rotation_matrix_*`, `supercell_cosets`, `supercell_structure`, `repeat_structure` | Rigid transforms and integer-supercell generation. |
| `closest_contact`, `layer_contact_report`, `self_image_contact`, `contact_report` | Measure physical contact distances after construction. |

Invariant: if a distance or contact is reported, it must be computed over a
complete set of relevant periodic images, not from one arbitrary unit-cell copy.

Primary tests include `tests/test_geometry.py`, `tests/test_plane_minimum_image.py`,
`tests/test_reduction.py`, `tests/test_contacts.py`, and
`tests/test_origin_invariance.py`.

## 2. Native Gram-form moire search

Main code:

- `src/cellstine/moire/search/gram_config.py`
- `src/cellstine/moire/search/gram_lattice.py`
- `src/cellstine/moire/search/gram_pairs.py`
- `src/cellstine/moire/search/gram_report.py`
- `src/cellstine/moire/search/gram.py`
- `src/cellstine/moire/search/results.py`
- `src/cellstine/moire/search/find.py`

Detailed notes:

- `src/cellstine/moire/MOIRE_SEARCH.md`
- `src/cellstine/moire/TWISTED_BILAYER.md`
- `docs/moire-performance.md`

Pipeline:

1. Normalize the supplied in-plane bases into a validated `SearchConfig`.
2. Reduce lattice gauges with Lagrange-Gauss style basis reduction.
3. Enumerate metric vectors and shell-bounded reduced bases.
4. Fold by proper layer symmetries; mirrors are not merged by default.
5. Join top and bottom candidate tables with the Gram-band/Löwner acceptance
   condition implied by the strain budgets.
6. Compute principal stretches, twist angle, polar rotation, affine layer maps,
   and optimal strain sharing.
7. Canonicalize pair keys and choose a deterministic Pareto-ranked shortlist.
8. Persist only schema-versioned JSON.

Key functions and objects:

| Code | Algorithmic role |
| --- | --- |
| `SearchConfig`, `SearchResult` | Validated physical search contract. |
| `_reduce_basis`, `_gram_triples`, `_point_group`, `_proper_subgroup` | Gauge reduction, metric triples, and symmetry preparation. |
| `_vector_table`, `_fold_sublattices`, `_fold_bases` | Build reduced candidate tables with symmetry folding. |
| `_loewner_mask`, `_join_candidates` | Vectorized exact candidate acceptance in Gram space. |
| `_stretches_from_gram`, `_twist_angles` | Recover physical stretch and twist from accepted Gram pairs. |
| `_canonical_pair_keys`, `_pair_orbit_keys` | Deterministic de-duplication under allowed symmetries. |
| `_pareto_front` | Deterministic non-dominated candidate selection. |
| `_affine_geometry`, `_finalize` | Compute recorded affine transforms and final shared lattice. |
| `build_results_document`, `validate_results`, `read_results` | JSON result schema and validation. |
| `run_find` | CLI/API-facing bilayer search orchestration. |

Invariant: builders consume recorded JSON affine transforms; they do not
recreate a legacy angle search.

Primary tests include `tests/test_moire_search_kernels.py`,
`tests/test_moire_theory.py`, `tests/test_moire_results_schema.py`,
`tests/test_moire_reference_sweep.py`, `tests/test_moire_symmetric_branch.py`,
and `tests/test_moire_mos2_reference.py`.

## 3. Moire construction and transforms

Main code:

- `src/cellstine/moire/builder/generator.py`
- `src/cellstine/moire/builder/make.py`
- `src/cellstine/moire/transform/translate.py`
- `src/cellstine/moire/moire.py`

Key functions and objects:

| Code | Algorithmic role |
| --- | --- |
| `LayerStack`, `GeneratedSupercell` | Construction records for one accepted bilayer candidate. |
| `_coset_representatives` | Enumerate integer cosets for supercell replication. |
| `_transform_layer_atoms`, `_recorded_layer_geometry` | Apply recorded affine layer transforms in-plane. |
| `_build_final_lattice`, `_finalise_cartesian_atoms` | Assemble the common lattice and Cartesian atoms. |
| `generate_from_results`, `generate_many_from_results` | Build one or many candidates from JSON. |
| `translate` | Shift an existing top layer or selected layer group in-plane. |
| `Moire.find`, `Moire.make`, `Moire.translate`, `Moire.visualize` | Public workflow methods. |

Invariant: top and bottom transformed supercell lattices must agree within the
configured tolerance before the structure is written.

Primary tests include `tests/test_moire_builder.py`,
`tests/test_moire_built_geometry.py`, `tests/test_layer_heights.py`, and
`tests/test_pipeline_interop.py`.

## 4. N-layer moire stack search

Main code:

- `src/cellstine/moire/search/nlayer.py`
- `src/cellstine/moire/builder/nlayer.py`
- `src/cellstine/moire/supermoire.py`
- `src/cellstine/moire/transform/translaten.py`

Current status: the implementation and tests exist, but the public release
contract still needs a decision. Earlier release notes describe N-layer as
deferred, while the current CLI exposes `moire stack-search` and
`moire stack-build`.

Algorithmic design:

1. Hold the base layer rigid.
2. Run the bilayer Gram engine once for each upper layer against the base.
3. Represent every accepted base-upper match as an integer base sublattice.
4. Intersect the base sublattices exactly with integer kernels.
5. Lift each layer through quotient matrices into the shared cell.
6. Apply a common unimodular reduction and revalidate geometry.
7. Build only from validated N-layer JSON results.

Key functions and objects:

| Code | Algorithmic role |
| --- | --- |
| `integer_left_kernel` | Exact integer kernel used for sublattice intersections. |
| `sublattice_intersection`, `intersect_sublattices` | Compute the shared integer base sublattice. |
| `quotient_matrix` | Lift per-layer cells into the common base cell. |
| `reduce_supercell` | Common unimodular reduction of the stack cell. |
| `LayerMatch`, `NLayerCandidate` | Typed records for pair matches and full-stack candidates. |
| `combine_layer_matches`, `viable_combinations` | Compose per-layer matches under atom/shape bounds. |
| `run_findn`, `read_nlayer_results` | Search orchestration and JSON reading. |
| `build_nlayer_supercell` | Construct a selected N-layer stack. |

Decision needed: call this experimental and document it, or hide/reject it for
the next public release.

## 5. Surface generation and adsorption sites

Main code:

- `src/cellstine/interface/surface/surface_cell.py`
- `src/cellstine/interface/surface/surface_supercell.py`
- `src/cellstine/interface/surface/surface_sites.py`
- `src/cellstine/interface/surface/stacking.py`
- `src/cellstine/interface/surface/termination.py`
- `src/cellstine/interface/surface/backend.py`

Detailed notes:

- `src/cellstine/interface/surface/SURFACE_CELLS.md`

Pipeline:

1. Parse Miller notation and derive the reciprocal surface normal.
2. Find primitive in-plane vectors for the requested plane.
3. Construct a canonical surface frame with the active plane in `xy`.
4. Group atomic planes by height and species.
5. Choose layer counts and vacuum conventions.
6. Apply in-plane repeats or explicit 2x2 surface matrices.
7. Detect adsorption site families on the selected surface side.

Key functions and objects:

| Code | Algorithmic role |
| --- | --- |
| `_primitive_surface_vectors`, `_select_surface_pair` | Find valid in-plane Miller-plane basis vectors. |
| `_surface_coordinate_frame` | Build the canonical slab coordinate frame. |
| `_build_native_primitive_surface_cell` | Generate the primitive slab-supporting cell. |
| `analyse_primitive_surface`, `build_surface_structure` | Preview and build surface structures. |
| `_find_bridge_sites`, `_classify_hollow`, `_site_from_uv` | Detect and classify adsorption sites. |
| `analyse_stacking`, `group_layers` | Infer close-packed stacking sequence and layer groups. |
| `termination_report` | Explain exposed species/termination. |
| `Surface.surface`, `Surface.sites` | Public surface workflow methods. |

Primary tests include `tests/test_surface.py`, `tests/test_surface_lattices.py`,
`tests/test_surface_plane_basis.py`, `tests/test_surface_stacking_sequence.py`,
and `tests/test_surface_termination.py`.

## 6. Interface matching and stacking

Main code:

- `src/cellstine/interface/workflow/lattice_match.py`
- `src/cellstine/interface/workflow/assembly.py`
- `src/cellstine/interface/workflow/interface.py`
- `src/cellstine/interface/surface/registry.py`
- `src/cellstine/interface/INTERFACE_STACKING.md`

Pipeline:

1. Generate or read bottom/top slab cells.
2. Search integer in-plane supercell matches under length, strain, and atom
   limits.
3. Rank candidates by strain, atom count, then surface area.
4. Enumerate registry/contact options while removing equivalent stackings.
5. Build the final slab-on-slab structure with controlled gap/vacuum rules.

Key functions and objects:

| Code | Algorithmic role |
| --- | --- |
| `MatchRequest`, `SlabPair` | Interface match input state. |
| `search_slab_pair`, `match_entries`, `sort_matches` | Integer lattice matching and deterministic ranking. |
| `build_match_document`, `validate_matches`, `read_matches` | JSON schema for match results. |
| `enumerate_registry_options`, `canonical_configuration` | Registry enumeration modulo allowed stacking equivalences. |
| `stack_structures` | Assemble the final interface structure. |
| `Interface.match`, `Interface.build`, `Interface.registries` | Public interface workflow methods. |

Primary tests include `tests/test_interface_match.py`,
`tests/test_interface_stacking.py`, `tests/test_interface_stacking_build.py`,
and `tests/test_lattice_match_ordering.py`.

## 7. Adsorbate placement, movement, and assembly

Main code:

- `src/cellstine/adsorbate/placement/operations.py`
- `src/cellstine/adsorbate/placement/place.py`
- `src/cellstine/adsorbate/transform/move.py`
- `src/cellstine/adsorbate/assemble.py`
- `src/cellstine/adsorbate/adsorbate.py`
- `src/cellstine/adsorbate/molecule.py`

Pipeline:

1. Classify the substrate as slab/surface/patch/bulk as requested.
2. If needed, build a surface from a bulk input before placement.
3. Detect available adsorption sites on the relevant surface side.
4. Frame the molecule, rotate rigidly around its center of mass, and optionally
   reframe axes.
5. Translate the closest molecule atom so it lies the requested height above
   the selected site.
6. Validate substrate/molecule contacts and self-image separation.
7. For `assemble`, search a synthetic molecular target lattice against a
   substrate using Gram-form matching.

Key functions and objects:

| Code | Algorithmic role |
| --- | --- |
| `center_of_mass_cartesian` | Mass-weighted molecule center. |
| `identify_top_group`, `identify_top_molecule` | Partition molecule/substrate groups in existing structures. |
| `_estimate_inplane_repeats_for_molecule` | Find a minimal substrate patch large enough for a molecule. |
| `_translate_molecule_to_site` | Enforce the physical molecule-to-surface gap. |
| `_combine_substrate_and_molecule` | Preserve ordering and selective dynamics while joining structures. |
| `place`, `move`, `assemble` | Workflow engines for placement, movement, and molecular assembly. |

Primary tests include `tests/test_adsorbate.py`,
`tests/test_molecule_framing.py`, and `tests/test_pipeline_interop.py`.

## 8. Defects, voids, and migration paths

Main code:

- `src/cellstine/defect/analysis.py`
- `src/cellstine/defect/sites.py`
- `src/cellstine/defect/generation.py`
- `src/cellstine/defect/supercell.py`
- `src/cellstine/core/voids.py`
- `src/cellstine/core/planar_voids.py`
- `src/cellstine/core/pathway.py`
- `src/cellstine/core/path_stage.py`

Detailed notes:

- `src/cellstine/defect/INTERSTITIAL_SITES.md`
- `src/cellstine/core/MIGRATION_PATH.md`

Pipeline:

1. Detect whether the structure behaves as bulk, slab/surface, or
   molecule-on-substrate.
2. Partition layers when slab-aware grouping is more appropriate than 3D bulk
   symmetry.
3. Group equivalent atom, interstitial, and adatom sites.
4. Generate requested defect structures from stable site IDs.
5. Choose defect supercells by image-separation and HNF enumeration.
6. Build migration paths by solving a species-preserving minimum-image atom
   assignment problem.

Key functions and objects:

| Code | Algorithmic role |
| --- | --- |
| `DefectSite`, `DefectAnalysis` | Stable defect-site and analysis records. |
| `_detect_structure_kind`, `_symmetry_groups_for_points` | Select the correct equivalence model. |
| `find_void_sites`, `find_planar_voids` | Bulk and planar void/interstitial candidates. |
| `hermite_normal_forms_3d`, `hermite_normal_forms_2d` | Enumerate candidate defect supercells. |
| `image_distance_of`, `SupercellChoice` | Score defect-image separation. |
| `DefectGenerationMixin` | Generate vacancy, substitution, interstitial, and adatom structures. |
| `optimal_assignment`, `match_atoms`, `build_migration_path` | Atom mapping and path interpolation. |

Primary tests include `tests/test_defect.py`, `tests/test_defect_supercell.py`,
`tests/test_voids.py`, `tests/test_voids_saddles.py`, and
`tests/test_defect_path.py`.

## 9. Symmetry, reciprocal lattices, k-points, and k-paths

Main code:

- `src/cellstine/core/symmetry2d.py`
- `src/cellstine/core/symmetry3d.py`
- `src/cellstine/core/pointgroup3d.py`
- `src/cellstine/core/reciprocal.py`
- `src/cellstine/core/brillouin.py`
- `src/cellstine/core/kpath.py`
- `src/cellstine/core/strata.py`
- `src/cellstine/symmetry/models.py`
- `src/cellstine/symmetry/reporting.py`
- `src/cellstine/symmetry/records.py`
- `src/cellstine/symmetry/spglib_adapter.py`
- `src/cellstine/symmetry/symmetry.py`
- `src/cellstine/symmetry/kpath_stage.py`

Detailed notes:

- `src/cellstine/core/RECIPROCAL.md`
- `src/cellstine/core/KPATH.md`

Key functions and objects:

| Code | Algorithmic role |
| --- | --- |
| `lattice_point_group`, `layer_point_group`, `idealised_layer_lattice` | Native 2D symmetry detection and layer idealisation. |
| `symmetry_operations`, `equivalent_atom_map`, `primitive_cell` | Native 3D operation scans and atom orbits. |
| `point_group_symbol`, `crystal_system_of_point_group` | Bare-lattice point-group classification. |
| `reciprocal_lattice`, `mesh_divisions_for_spacing`, `mesh_points` | Reciprocal basis and KPOINTS mesh generation. |
| `wigner_seitz_cell`, `brillouin_zone`, `zone_boundary_distance` | Brillouin-zone geometry. |
| `special_points`, `band_path`, `segment_strata` | Special-point naming, path derivation, and segment classification. |
| `Symmetry.analyse`, `Symmetry.reduce`, `Symmetry.kpoints`, `Symmetry.kpath` | Public symmetry workflow methods. |

Primary tests include `tests/test_symmetry_models.py`,
`tests/test_symmetry_records.py`,
`tests/test_symmetry_spglib_adapter.py`,
`tests/test_symmetry_workflow.py`,
`tests/test_symmetry2d.py`, `tests/test_symmetry3d.py`,
`tests/test_reciprocal.py`, `tests/test_kpoints_layout.py`, and
`tests/test_kpath.py`.

## 10. CLI and workflow orchestration

Main code:

- `src/cellstine/cli/spec.py`
- `src/cellstine/cli/plain.py`
- `src/cellstine/cli/plain_moire.py`
- `src/cellstine/cli/plain_adsorbate.py`
- `src/cellstine/cli/plain_defect.py`
- `src/cellstine/cli/plain_surface.py`
- `src/cellstine/cli/plain_interface.py`
- `src/cellstine/cli/plain_symmetry.py`
- `src/cellstine/cli/plain_view.py`
- `src/cellstine/cli/rich_app.py`
- `src/cellstine/cli/main.py`
- `src/cellstine/cli/interactive/`
- `src/cellstine/core/base.py`
- `src/cellstine/core/manifests.py`
- `src/cellstine/core/report.py`

Key functions and objects:

| Code | Algorithmic role |
| --- | --- |
| `resolve_moire_strains`, `parse_twist_window` | Public CLI parameter normalization. |
| `build_parser` | Dependency-free stdlib command grammar. |
| `RichGuidedUI`, `PlainGuidedUI`, `run_interactive` | Guided workflow presentation and command-preview/run-confirmation loop. |
| `execute_namespace`, `dispatch_namespace`, `main` | CLI dispatch into workflow classes. |
| `Base.create_run_dir`, `Base.write_manifest`, `CommandResult` | Consistent workflow result recording. |
| `format_result` | Shared terminal report formatting. |

Invariant: guided mode must preview the generated command and ask before
executing it. Rich/Typer are optional; base installs must remain NumPy-only.

Primary tests include `tests/test_cli_rework.py`,
`tests/test_cli_startup.py`, `tests/test_cli_end_to_end.py`, and
`tests/test_cli_interactive_moire.py`.

## Near-term technical debt

1. Decide and document N-layer moire status.
2. Split the largest modules only after preserving exact imports and tests.
3. Move tests into domain subfolders as a pure file move.
4. Keep all generated scientific output ignored; track only curated examples.
5. Replace stale command references whenever the CLI public contract changes.
