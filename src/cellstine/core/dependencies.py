"""Optional dependency discovery and backend selection."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from importlib.util import find_spec
from typing import Dict


@dataclass(frozen=True)
class DependencyStatus:
    """Availability and version metadata for a runtime dependency."""

    name: str
    available: bool
    version: str | None


class DependencyManager:
    """Resolve optional dependencies and choose implementation backends."""

    _PACKAGES = {
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "plotly": "plotly",
        "pymatgen": "pymatgen",
        "spglib": "spglib",
    }

    def __init__(self) -> None:
        self._cache: Dict[str, DependencyStatus] = {}

    def status(self, name: str) -> DependencyStatus:
        key = str(name)
        if key in self._cache:
            return self._cache[key]
        package_name = self._PACKAGES.get(key, key)
        available = find_spec(package_name) is not None
        version = None
        if available:
            try:
                version = metadata.version(package_name)
            except metadata.PackageNotFoundError:
                version = None
        result = DependencyStatus(name=key, available=available, version=version)
        self._cache[key] = result
        return result

    def has(self, name: str) -> bool:
        return bool(self.status(name).available)

    def versions(self) -> Dict[str, str]:
        versions: Dict[str, str] = {}
        for name in self._PACKAGES:
            status = self.status(name)
            if status.available and status.version:
                versions[name] = status.version
        return versions

    def choose_backend(self, requested: str = "auto", *, feature: str | None = None) -> str:
        choice = str(requested or "auto").lower()
        if choice not in {"auto", "native", "pymatgen"}:
            raise ValueError(f"unsupported backend '{requested}'")
        if choice == "native":
            return "native"
        if choice == "pymatgen":
            if not self.has("pymatgen"):
                details = f" for {feature}" if feature else ""
                raise RuntimeError(f"pymatgen is required{details}, but it is not installed")
            return "pymatgen"
        return "pymatgen" if self.has("pymatgen") else "native"

    def choose_symmetry_backend(self, requested: str = "auto", *, feature: str | None = None) -> str:
        """Resolve the crystallographic symmetry backend.

        This intentionally prefers direct spglib over pymatgen. pymatgen remains
        available to the IO converter for broad formats, but symmetry workflows
        should not route through it.
        """

        choice = str(requested or "auto").lower()
        if choice not in {"auto", "native", "spglib"}:
            raise ValueError(f"unsupported symmetry backend '{requested}'")
        if choice == "native":
            return "native"
        if choice == "spglib":
            if not self.has("spglib"):
                details = f" for {feature}" if feature else ""
                raise RuntimeError(f"spglib is required{details}, but it is not installed")
            return "spglib"
        return "spglib" if self.has("spglib") else "native"
