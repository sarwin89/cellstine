"""Species-label helpers shared across workflows."""

from __future__ import annotations

from typing import List, Sequence, Tuple


def expand_species(
    species: Sequence[str],
    counts: Sequence[int],
    fallback: str | None = None,
    *,
    natoms: int | None = None,
    padding: str = "X",
) -> List[str]:
    """Expand POSCAR species/count records into one label per atom.

    ``fallback`` supplies a symbol for files that carry counts but no species
    line.  When ``natoms`` is given the result is forced to that length, padded
    with ``padding`` and truncated if need be, which is what a renderer wants:
    it must label every atom it draws and cannot fail on a malformed header.
    Leave ``natoms`` unset where a mismatch is an error worth raising.
    """

    if species:
        labels = [str(symbol) for symbol in species]
    elif fallback is not None:
        labels = [str(fallback)] * len(counts)
    else:
        raise ValueError("POSCAR species labels are required")

    expanded: List[str] = []
    for symbol, count in zip(labels, counts):
        expanded.extend([symbol] * int(count))

    if natoms is None:
        return expanded
    total = int(natoms)
    if len(expanded) < total:
        expanded.extend([str(padding)] * (total - len(expanded)))
    return expanded[:total]


def group_species(labels: Sequence[str]) -> Tuple[List[str], List[int], List[int]]:
    """Turn one label per atom into a POSCAR species header and an atom order.

    This is the inverse of :func:`expand_species`, and the reordering is the
    point of it.  A POSCAR names its species once and then gives *all* the atoms
    of the first species, then all of the second, so a file that lists its atoms
    in any other order -- an XYZ of ``C O C O``, or a CIF that interleaves its
    sublattices -- has to be permuted as well as counted.  Counting alone, with
    the coordinates left where they were, silently renames atoms.

    The returned triple is the species in order of first appearance, how many
    atoms each of them has, and the permutation that brings the atoms into that
    order; apply the permutation to every per-atom array.
    """

    order: List[str] = []
    grouped: dict[str, List[int]] = {}
    for index, label in enumerate(labels):
        symbol = str(label)
        bucket = grouped.get(symbol)
        if bucket is None:
            bucket = grouped[symbol] = []
            order.append(symbol)
        bucket.append(index)
    counts = [len(grouped[symbol]) for symbol in order]
    positions = [index for symbol in order for index in grouped[symbol]]
    return order, counts, positions
