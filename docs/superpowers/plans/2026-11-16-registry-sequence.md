# Registry reporting and plane sequence correctness

Base: dev `3a9afc0`, with its exact-head CI and downloaded coverage closed.
Work only in this isolated checkout; preserve the shared checkout and run data.

1. Add regressions for the greedy plane-matching false negative, Hall deficits,
   duplicate multiplicity, periodic boundaries and partial repeating prefixes.
2. Preserve ordinary nearest-first matches and repair failures through an exact,
   iterative augmenting path, without a dense pair matrix. Optimize prefix search
   only with predecessor timing and memory evidence.
3. Measure actual slab boundary registry in a common primitive gauge. Preserve
   the public label-only summary helper; represent nonideal offsets explicitly.
   Test translated 60/120-degree contacts, active and passive paths, tolerance
   boundaries and written metadata. Keep geometry and public signatures stable.
4. Inspect all bodies in registry, sequence, sites and assembly (28 existing plus
   added helpers); record evidence, including the 26-letter format limitation
   and any unproven interchangeability assumptions without speculative changes.
5. Benchmark predecessor/current; obtain one independent read-only batch review.
   Freeze source, run focused/full coverage and package/installed checks once,
   update the source-hashed ledger, commit scoped files and push dev. Close all
   supported-version exact-head CI and inspect downloaded coverage before the
   next runtime batch. No publishing, tags or package uploads.
