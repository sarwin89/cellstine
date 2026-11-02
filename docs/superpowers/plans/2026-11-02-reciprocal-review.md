# Reciprocal mesh review: next owned batch

Inspect all 26 tracked functions in `core/reciprocal.py`, with KPOINTS and symmetry callers. This is Codex's next scope; Claude owns concurrent root moiré/search/CLI work. This plan records initial inspection during the frozen Part04 checkpoint. It does not change source or close any pending review row.

## Demonstrated starting finding

The advertised collision-free `_encode` uses an overflowing int64 polynomial. For denominator 2097152, the distinct mesh numerators `[-1048576,0,0]` and `[0,0,0]` both produce 8796095119360. Both points occur in the valid `(1048576,1,1)` mesh. The two-row reproduction avoids allocating the full mesh. Investigate exact, deterministic ordering/key alternatives with a fast ordinary-range path and prove orbit membership/representative ordering, not merely key uniqueness.

The two-row orbit probe with only identity and time reversal disabled returns `[0,0]` instead of separate `[0,1]` labels, demonstrating the scientific consequence rather than a harmless private key representation change. Add a focused exact orbit regression and a resource benchmark for a realistic large grid; avoid multiplying a million-point case across every ordinary CI test unnecessarily.

Initial spacing probe: `mesh_divisions_for_spacing(np.eye(3), 1e-20)` emits a cast warning and returns `(1,1,1)`, whose actual step is approximately 6.283 rather than the requested tiny step. The request exceeds the current integer division range; it must produce a clear supported-range error rather than a plausible undersampled mesh.

## Inspect together

- Reciprocal row-vector convention, volume and spacing guarantees, minimum divisions, supported numeric domains and integer conversion. Refuse unrepresentable requested meshes clearly rather than silently returning undersampling.
- Numerator construction and wrapping: avoid three retained meshgrid arrays plus the stacked output when direct preallocation/broadcasting preserves exact iteration order. Check denominator arithmetic before allocation.
- Integer rotation conversion/inversion, finite point-group closure, mesh-preserving subgroup filtering, reciprocal action and time reversal. Avoid silently changing existing scientific tolerances; identify actual unsupported/malformed input separately.
- Orbit labels and representative ordering: verify complete symmetry images, exact weights summing to full count, shifted/anisotropic meshes and identities.
- `KpointMesh` properties/summary and diagonal supercell divisions. Keep unsupported general-supercell policy explicit rather than implementing a new scientific feature.

Prepare focused regressions only for actual findings, benchmark time/peak allocation against the next committed dev checkpoint, obtain one whole-batch independent review, then one frozen full/package checkpoint. Preserve scientific data and APIs; do not integrate root files wholesale. Do not mark a row verified from source reading or coverage alone. No publication.
