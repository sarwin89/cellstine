# Architecture

CELLSTINE uses a `src/` package layout.

```text
src/cellstine/
  cli/          argparse entrypoints plus interactive command-building flows
  core/         shared models, manifests, lattice helpers, transforms, validation
  io/           VASP I/O, conversion, orientation, and registry code
  moire/        moire workflows, search engines, builders, and transforms
    search/     angle, lattice, commensuration, and finder engines
    builder/    supercell generation and make stages
    transform/  rigid and layer-wise structure transforms
  adsorbate/    molecule-on-substrate workflows
    placement/  site placement and molecule/substrate assembly helpers
    transform/  molecule movement helpers
  interface/    interface workflows and the canonical surface package
    surface/    slab generation, adsorption sites, and surface reports
    workflow/   interface build and match orchestration
  defect/       defect records, analysis, generation, and workflow class
  symmetry/     equivalent-site and cell-reduction workflows
  visualize/    workflow class, rendering backends, and result visualizers
    backends/   matplotlib and Plotly structure renderers
    results/    result-file visualization builders
```

## Ownership Rules

- Generic math, coordinate transforms, species expansion, manifests, and validation belong in `core`.
- File formats and structure conversion belong in `io`.
- Molecule placement and COM-based movement belong in `adsorbate`.
- Slab generation and adsorption-site detection belong in `interface/surface`; adsorbate workflows should import them from there rather than duplicating surface logic.
- Moire angle search and commensuration belong in `moire/search`; supercell construction belongs in `moire/builder`; layer transforms belong in `moire/transform`.
- Visualization backends belong in `visualize/backends`; result-file visualizers belong in `visualize/results`.
- Workflow modules may reuse shared engines, but should not shell through another workflow for unrelated domain logic.

## Public API

Top-level imports expose workflow classes and public records:

```python
from cellstine import Moire, Adsorbate, Molecule, Interface, Surface, Defect, Symmetry, Visualize
```
