"""Public package API for CELLSTINE."""

from importlib import import_module

_PUBLIC_EXPORTS = {
    "Adsorbate": ("cellstine.adsorbate.adsorbate", "Adsorbate"),
    "Defect": ("cellstine.defect.workflow", "Defect"),
    "DefectAnalysis": ("cellstine.defect.records", "DefectAnalysis"),
    "DefectSite": ("cellstine.defect.records", "DefectSite"),
    "DependencyManager": ("cellstine.core.dependencies", "DependencyManager"),
    "EquivalentAtomGroup": ("cellstine.symmetry.symmetry", "EquivalentAtomGroup"),
    "Interface": ("cellstine.interface.workflow.interface", "Interface"),
    "Molecule": ("cellstine.adsorbate.molecule", "Molecule"),
    "Moire": ("cellstine.moire.moire", "Moire"),
    "RunManifest": ("cellstine.core.manifests", "RunManifest"),
    "StructureConverter": ("cellstine.io.converters", "StructureConverter"),
    "Supermoire": ("cellstine.moire.supermoire", "Supermoire"),
    "Surface": ("cellstine.interface.surface.surface", "Surface"),
    "Symmetry": ("cellstine.symmetry.symmetry", "Symmetry"),
    "SymmetryAnalysis": ("cellstine.symmetry.symmetry", "SymmetryAnalysis"),
    "SymmetryOperation": ("cellstine.symmetry.symmetry", "SymmetryOperation"),
    "VaspIO": ("cellstine.io.vasp", "VaspIO"),
    "Visualize": ("cellstine.visualize.visualize", "Visualize"),
    "VisualizationRun": ("cellstine.visualize.results.plotly", "VisualizationRun"),
}

__all__ = [
    "Adsorbate",
    "Defect",
    "DefectAnalysis",
    "DefectSite",
    "DependencyManager",
    "EquivalentAtomGroup",
    "Interface",
    "Molecule",
    "Moire",
    "RunManifest",
    "StructureConverter",
    "Supermoire",
    "Surface",
    "Symmetry",
    "SymmetryAnalysis",
    "SymmetryOperation",
    "VaspIO",
    "Visualize",
    "VisualizationRun",
]

__version__ = "4.0.0"


def __getattr__(name: str):
    """Load public workflow classes only when they are requested."""
    if name not in _PUBLIC_EXPORTS:
        raise AttributeError(f"module 'cellstine' has no attribute {name!r}")
    module_name, attr_name = _PUBLIC_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
