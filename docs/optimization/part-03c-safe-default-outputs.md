# Part 03c: reserve actual default output paths

Baseline: `a4a3a0ec4d24b782f79c1249659da1d0f459ec6d`. Its seven CI jobs passed in run `41398913199`; Python 3.12 reported 3,257 passed/four skipped and a successful ledger check. The expanded minimum lane installed NumPy 1.26.4 and passed 257 tests. Unexpired coverage artifact `13120128556` records that exact head, digest `e447a590eb9292ffdc988b3cf2e651c333cf251ba5b4daaf6ac51e90f671e8f0`.

## Problem and resulting behavior

Different input labels, or run100, can produce the same legacy filename suffix. `Visualize.structure` consequently wrote both results to `structure_viewer_TIMESTAMP.html`. The second run replaced the first artifact even though their run directories were distinct. Expanding numeric parsing would also reinterpret ordinary Miller labels such as111.

Default destinations now use exclusive filesystem reservation. A free preferred filename is unchanged. An occupied filename gets `_02`, `_03`, ... before its final extension; a default batch directory gets the same numbered fallback. The legacy suffix parser retains its values. Explicit file paths keep their existing overwrite behavior; choosing an explicit directory still leaves generated filenames automatic, so those files are reserved individually.

For example, two same-minute default visualizations now retain `structure_viewer_261102-1200.html` and `structure_viewer_261102-1200_02.html`. Writer-returned/selected paths flow into results and manifests. No existing file, directory or symlink is deleted or deliberately replaced by reservation.

## Individually inspected scope

The 25 changed original bodies below were inspected with their callers, return schemas and scientific delegation. The previously inspected legacy parser is reconsidered without changing its executable body. Two new helper functions complete this scope. Independent review read all26 originals and both helpers statically; its report explicitly does not claim an independent runtime rerun.

| Source | Functions | Preserved behavior and allocation change |
| --- | --- | --- |
| `core/outputs.py` | New `reserve_default_file`, `reserve_default_directory` | Exclusive leaf creation, unchanged preferred name, numbered collisions, parent creation and propagated errors. |
| `core/base.py` | `run_output_suffix` | Executable body unchanged; documentation clarifies historical syntax. Actual reservation supplies default-output identity. |
| `adsorbate/adsorbate.py` | `Adsorbate._expand_slab_substrate`, `.place`, `.move` | Preserve substrate preparation, transforms, contacts, metadata and explicit paths; reserve shared/default destinations. Run-owned substrate intermediates keep their names. |
| `adsorbate/placement/operations.py` | `transform_top_molecule`, `place_molecule_on_site`, `shift_top_layer` | Preserve molecular geometry/vacuum/site/flag calculations and output ordering; protect direct API defaults. Resolve the selected leaf after reservation. |
| `core/layers.py` | `shift_top_layer` | Preserve layer selection, shift convention and writer arguments; protect its direct default filename. |
| `defect/generation.py` | `DefectGenerationMixin.generate` | Preserve selected-site/replacement order, host/defect construction and contact data; reserve each generated filename and record its actual path. |
| `interface/surface/backend.py` | `build_surface` | Preserve slab/site calculations; reserve automatic slab and automatic sibling report, including an implicit sibling of an explicitly chosen slab. |
| `interface/surface/surface.py` | `Surface.surface`, `.sites` | Preserve descriptor, Miller convention, stacking/termination reports and explicit arguments; reserve default slab/site artifacts separately. |
| `interface/workflow/interface.py` | `Interface.build`, `._build_from_match` | Preserve matching/strain/registry/stacking/atom checks and result schemas; reserve default destination only. |
| `moire/builder/make.py`, `moire/builder/nlayer.py` | Each `generate_from_results` | Preserve index selection, geometry and species/flags; protect automatic candidate filenames even in explicit directories. |
| `moire/moire.py` | `Moire.make`, `.translate` | Reserve the default batch directory only when used; preserve explicit file/directory branches and post-generation suffix names. Protect translation defaults. |
| `moire/supermoire.py` | `Supermoire.maken` | Reserve default batch directory; preserve N-layer parameters, result order and filename suffixing inside the owned directory. |
| `symmetry/kpath_stage.py` | `BandPathMixin.kpath` | Preserve native path/point calculations, counts, labels, JSON metadata and explicit KPOINTS path; reserve default KPOINTS only. |
| `symmetry/symmetry.py` | `Symmetry.reduce`, `.lattice_reduce`, `.kpoints` | Preserve backend choice, crystallography, mesh calculations and serialization; reserve default destinations only. |
| `visualize/results/plotly.py` | `build_visualization` | Preserve frame selection/order and generated HTML; protect direct default destination. |
| `visualize/visualize.py` | `Visualize.moire_results`, `.structure` | Preserve matplotlib/HTML branches, direction reporting, selected inputs and manifests; reserve actual default paths. |

The other eight previously reviewed base functions have identical executable ASTs to the baseline; their existing status/evidence can be repinned after the completed-run hash check. Unchanged functions elsewhere in touched modules remain pending unless they already have independent inspection evidence. A changed module never silently inherits a complete review claim.

## Filesystem and compatibility decisions

`open('xb')` and leaf `mkdir()` arbitrate competing writers. Parents are created outside retry loops. On Windows, an occupied directory at a preferred filename raises `PermissionError`; retry is allowed only when that leaf is confirmed as a directory. Genuine permission/parent errors propagate. Leaf paths are not resolved before reservation, so an existing symlink is treated as occupied instead of redirecting the fallback filename to its target directory.

Public workflows pass selected paths explicitly to lower-level writers, avoiding double reservation. Direct lower-level API defaults reserve their own paths. Batch suffix renames operate inside a fresh reserved default directory; explicit directories rely on per-file reservation and retain their old rename policy. Duplicate selected candidate indices keep separate generated files and their original result-list order.

Reservation creates an owned empty file/directory before writing. A failed validation, optional dependency or writer can leave that new empty/partial artifact. This is observable and explicitly retained as a nontransactional design decision; existing scientific data stays protected. Reservation also does not defend against an external actor deleting/replacing the selected path during the write.

This resolves the inspected legacy-parser deferral for its default-output consumers. It is not universal protection of every library writer: low-level search APIs still treat `output_root` as a reusable working directory, including `results.json`. Explicit file destinations also retain their previous behavior. Those contracts remain visible for the I/O/search reviews.

## Focused evidence and review

The initial regression batch reported seven failures/one pass, including two actual public visualization overwrites. Shared helpers were absent at that point. The occupied-directory test then reproduced Windows `PermissionError`; the narrow guard fixed it while a separate test preserves genuine permission failures.

One combined affected-caller selection passed **210 tests**, with13 existing warnings, in36.73seconds. It includes visualization, adsorbates, surfaces, interfaces, symmetry, bilayer/N-layer builders and defect supercells. Three additional targeted integration regressions passed in0.73seconds: public batch-directory reuse, repeated candidates in an explicit directory and surface sibling-report collision. There are13 new tests in total. No broad suite was rerun for those three additions.

Independent review found no blocking issue after reading all26 originals, both helpers and relevant downstream writers. It confirmed path/manifest propagation, explicit-path branches, batch renames and surface pairs. Symlink behavior was assessed statically rather than claimed as an independent runtime test. The reviewer ran no duplicate suite or build. This is a correctness/data-preservation change; no timing speedup or artificial performance benchmark is claimed.

## Checkpoint gates

The single full coverage checkpoint passed **3,270 tests, four skipped,81 warnings in857.74seconds**. The skips retain the absent external Lean references and existing surface-period case. All250 frozen Python hashes and the complete Python file set match after terminal success. Base executable ASTs match the baseline. Canonical ledger drift and Ruff F checks pass.

Coverage is **12,462/14,352 statements (86.83%)** and **3,238/4,512 branches (71.76%)**. The new reservation module has29/29 statements and2/2 branches covered; this is not a claim about adversarial filesystem replacement or every platform condition.

The one wheel/sdist build and Twine metadata checks pass. All141 wheel package modules match source bytes; the source distribution includes the new helper/tests and excludes scientific runs/environments. Installed core-only, CLI and source-distribution environments under NumPy2.5.4 each passed concurrent reservations, occupied paths, public cross-label visualization, explicit-path behavior, actual manifest paths, help and dependency checks. Final documentation/ledger support bytes are compared after this report update; no repeated install is needed for documentation-only archive refresh.

The reconciled ledger contains **1,103 functions:175 reviewed unchanged,66 optimized verified,two investigated deferrals and860 pending**. Thus **243 functions have inspection evidence**. The 25 newly inspected original bodies and two new helpers add27 records of evidence. The previously inspected parser becomes reviewed unchanged after the consumer-level fix; the eight other base records retain their previous status with executable-AST proof and current dependent verification. Untouched pending functions remain pending.

The minimum NumPy lane additionally includes the13 new reservation tests, for270 tests. Its actual result and all supported-version/package/coverage gates must still be read from the pushed commit before claiming the checkpoint remotely verified.

While that checkpoint runs, the next I/O inspection has identified two separate pre-existing integrity risks in `StructureConverter`: fallback molecular conversion writes then deletes `source.with_suffix('.tmp.xyz')`, and `_read_xyz` can accept fewer atoms than its header declares. These are queued for the next code batch and are not claimed fixed or counted as completed function reviews here. Publication remains unauthorized.
