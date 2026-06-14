# Architecture

CELLSTINE uses a `src/` package layout.

```text
src/cellstine/
  cli/          argparse and guided interface entrypoints
  core/         shared models, manifests, lattice helpers, transforms, validation
  io/           VASP I/O, conversion, orientation, and registry code
  moire/        moire search/build/translate workflows
  adsorbate/    molecule-on-substrate placement, movement, and assembly
  interface/    slab generation, adsorption sites, and interface building
  defect/       defect analysis and generation
  symmetry/     equivalent-site and cell-reduction workflows
  visualize/    matplotlib-first and optional Plotly rendering
```

## Ownership Rules

- Generic math, coordinate transforms, species expansion, manifests, and validation belong in `core`.
- File formats and structure conversion belong in `io`.
- Molecule placement and COM-based movement belong in `adsorbate`.
- Slab generation and adsorption-site detection belong in `interface`.
- Moire search and supercell construction belong in `moire`.
- Workflow modules may reuse shared engines, but should not shell through another workflow for unrelated domain logic.

## Public API

Top-level imports expose workflow classes and public records:

```python
from cellstine import Moire, Adsorbate, Molecule, Interface, Surface, Defect, Symmetry, Visualize
```
