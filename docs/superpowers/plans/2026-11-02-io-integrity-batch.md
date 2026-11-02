# I/O integrity and resource batch

**Execution:** Autonomous inline work, following the user's instruction to batch code changes and reduce repeated verification. Inspect while previous CI runs; use focused regressions for actual findings, then one combined full/package checkpoint after the whole I/O batch is ready.

**Baseline:** Output-safety commit `c7dec6cc0d83d0183e9f9b438d0ba660b10954d6`, CI41405522888. Confirm its exact-head result before the next push.

**Scope:** All37 pending functions in `io/converters.py`9, `io/native.py`9, `io/kpoints.py`14, `io/orientation.py`3 and `io/vasp.py`2. Read declarative registry/exports too. The two `StructureRecord` functions already have inspection evidence and are not counted again.

## Code work

- Inspect every scoped function/caller and valid/malformed input policy before deciding to change it. Preserve format fields, species/atom order, scaling factors, flags, coordinate conventions and scientific tolerances. Leave sound functions unchanged with a reason.
- Fix molecular fallback temporary-file ownership: `_read_with_pymatgen` currently writes then deletes the source's sibling `.tmp.xyz`. Use a separately owned unique temporary file, close it before external writing on Windows, and clean up only that file on success or failure. Preserve source files and neighbouring data. Assess original source metadata and requested molecular vacuum propagation together.
- Make XYZ frame parsing enforce its declared atom count and reject malformed/nonfinite rows clearly. Read only the selected first frame, retaining the existing first-frame behavior for trajectories and stable species grouping. Preserve the bounding-box/vacuum convention for valid data.
- Inspect `_write_xyz` label expansion for a streaming replacement that preserves exact valid output and conversion/error ordering. Benchmark actual allocation changes against the trusted baseline; no CI timing thresholds.
- Complete native POSCAR and KPOINTS inspection in the same batch. Record and fix demonstrated integrity issues together where compatible. Treat exceptional numerical orientation/validation range as the existing end-to-end investigation; do not silently alter absolute thresholds or claim universal finite-range support.

## Meaningful regressions

- Fallback conversion preserves a pre-existing sibling `.tmp.xyz`, uses an owned temporary, and cleans it if the external writer fails. Returned source metadata and explicit vacuum must describe the actual conversion.
- A two-atom XYZ header with one atom raises instead of silently changing the molecule. Cover short/nonnumeric/nonfinite rows and retain valid interleaved-species/trajectory-first-frame behavior.
- Prepared long trajectory input has bounded first-frame parsing allocation; prepared large coherent records produce byte-identical XYZ output without expanding a full atom-label list.
- Reuse existing POSCAR roundtrip, distance-spectrum orientation and KPOINTS read/write tests for the completed batch; add cases only for demonstrated gaps. Do not rerun broad passing selections after documentation-only changes.

## One checkpoint

Finish the function ledger, obtain one independent review of the complete change, run one combined focused selection, freeze source/tests, then run one full coverage and package/install pass. Commit/push the complete I/O batch and inspect exact-head supported-version/minimum-dependency CI. Keep the full project goal active while any pending functions remain. No publication is authorized.
