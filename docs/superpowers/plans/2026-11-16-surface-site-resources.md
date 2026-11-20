# Surface-site memory and exact ordered deduplication

Base dev `8b04d03`; all16 exact-head CI43857657826 jobs and downloaded coverage
artifact14221503471 are closed before this batch's runtime edits.

1. Add literal regressions for rounded-prefilter losses, including the default
   tolerance's nontransitive chain. Preserve ordered greedy distance semantics
   and remove only exact duplicates before matching. Preallocate accepted storage.
2. Bound nearest-neighbor pair displacements and anchor/image distance
   temporaries without neighbor/image caps. Keep the dense helper's API, exact
   arithmetic, image ordering, exclusions and cutoff+1e-12. Preserve bridge
   anchor-major/image-major order when chunking; disclose expanded-image and
   output-size memory separately from bounded pair temporaries.
3. Compare ordinary outputs against predecessor and scalar periodic references;
   test empty/singleton/duplicates, skew cells, boundaries, ownership and real
   written site reports. Measure small/large CPU and allocation tradeoffs.
4. Complete individual reviews for surface_sites15, Surface workflow6 and
   build/match wrappers2, plus added helpers. Prepared source notes are navigation
   only. Investigate physical naming premises without speculative policy changes.
5. Obtain one independent whole-body review; freeze source, then one affected
   pass/full coverage/build/install batch. Promote ledger only after gates, refresh
   final support sdist, commit scoped files and push dev. Close exact-head CI and
   downloaded coverage before another runtime batch. Preserve shared main/Claude
   work and all scientific data. No release, tag or package upload.
