# Interface Workflow

The `surface` workflow generates slabs and analyses adsorption sites. The
`interface` workflow matches and builds slab-on-slab heterostructures.

## Common Commands

```bash
cellstine surface build input/examples/Au_Bulk.vasp --miller 111 --layers 4 --vacuum 15
cellstine surface sites output/examples/Au_Bulk_111_surface.vasp
cellstine interface build output/examples/Au_Bulk_111_surface.vasp output/examples/Au_Bulk_111_surface.vasp --gap 3.0
```

## Miller Notation

Miller indices accept comma-separated or compact forms:

```text
111
001
1,1,0
1,1,2x
```

`x` denotes a negative index, so `111x` means `(1, 1, -1)`.

## Notes

- Generated slabs are oriented with the active plane in `xy` and the surface normal along `c`.
- Vacuum is placed on the slab side used for adsorbate-style workflows.
- Site reports include available families such as top, bridge, hollow, fcc/hcp hollow, and fourfold hollow when detected.
