# Moire Search Performance

CELLSTINE's optimized moire search treats the twist angle as an output of vector matching rather than sweeping every angle on a fixed grid.

## Practical Guidance

- `nindex` controls the integer search range. Larger values can grow very quickly.
- Strain tolerances are fractions. For example, `0.001` is 0.1 percent and `0.02` is 2 percent.
- Wide strain bands at large `nindex` can create near-continuous angle families and many vector-pair candidates.
- Use `--workers` for larger searches when multiprocessing overhead is worth it.
- Use `--progress` when running expensive searches so timing stages are visible.

## Useful Commands

```bash
cellstine moire find input/examples/graph.vasp input/examples/graph.vasp --nindex 40 --angle-strain-tolerance 0.001 --workers 4 --progress
cellstine moire find input/examples/mos2.vasp input/examples/mos2.vasp --nindex 20 --max-pair-matches 2000000 --progress
```

The detailed implementation note also lives near the source in `src/cellstine/moire/MOIRE_SEARCH.md`.
