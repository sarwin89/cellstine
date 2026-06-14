"""Species-label helpers shared across workflows."""

from __future__ import annotations

from typing import List, Sequence


def expand_species(species: Sequence[str], counts: Sequence[int], fallback: str | None = None) -> List[str]:
    """Expand POSCAR species/count records into one label per atom."""

    if species:
        labels = [str(symbol) for symbol in species]
    elif fallback is not None:
        labels = [str(fallback)] * len(counts)
    else:
        raise ValueError("POSCAR species labels are required")

    expanded: List[str] = []
    for symbol, count in zip(labels, counts):
        expanded.extend([symbol] * int(count))
    return expanded
