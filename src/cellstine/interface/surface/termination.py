"""What a slab exposes, and whether it is still the compound it was cut from.

Cutting a compound along a Miller plane is arithmetic that always succeeds, but
two things about the result decide whether the cell is a usable model of a
surface, and neither of them is visible in the atom count:

*Stoichiometry.*  A slab of a compound need not hold a whole number of formula
units.  A five-layer cut of rocksalt along (1 0 0) does, a cut along (1 1 1)
does not: its planes alternate between the two species, so it always carries one
species in excess.  A total energy of such a cell cannot be compared with the
bulk energy without a chemical-potential reference for the excess atoms, and a
surface energy read off it is meaningless.

*The two faces.*  A slab has two of them, and only if they are the same
termination does the cell model one surface twice.  When they differ -- again
the (1 1 1) cut of rocksalt, one face all cations, the other all anions -- the
cell carries a dipole along the normal, the two faces interact through the
vacuum, and a plane-wave calculation needs a dipole correction (``IDIPOL`` and
``LDIPOL`` in VASP) or a symmetric slab before it means anything.  This is
Tasker's type III surface, and it is the classic way a slab calculation goes
quietly wrong.

Both are decided here by exact integer arithmetic on the species counts -- the
formula unit is the counts divided by their greatest common divisor, and the two
terminations are compared as multisets of species -- so the report never depends
on a tolerance except for the one that groups atoms into layers.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ...core.layers import LAYER_TOLERANCE as _LAYER_TOLERANCE, layer_partition
from ...core.species import expand_species
from ...core.vacuum import normal_heights

__all__ = [
    "TerminationReport",
    "formula_unit",
    "layer_species",
    "termination_report",
]

#: Height difference below which two atoms count as one layer, shared with the
#: rest of the package (``core.layers.LAYER_TOLERANCE``).
LAYER_TOLERANCE = _LAYER_TOLERANCE


def formula_unit(counts: Mapping[str, int]) -> dict[str, int]:
    """Return the smallest whole-number species ratio of ``counts``.

    ``{'Na': 4, 'Cl': 4}`` and ``{'Na': 108, 'Cl': 108}`` both reduce to
    ``{'Na': 1, 'Cl': 1}``, which is the unit a slab has to hold a whole number
    of to be stoichiometric.
    """

    values = {str(key): int(value) for key, value in counts.items() if int(value) > 0}
    if not values:
        return {}
    divisor = 0
    for value in values.values():
        divisor = math.gcd(divisor, value)
    divisor = max(divisor, 1)
    return {key: value // divisor for key, value in values.items()}


def _format_counts(counts: Mapping[str, int]) -> str:
    """Return a formula string such as ``Na1 Cl1``, in a stable order."""

    return " ".join(f"{symbol}{int(counts[symbol])}" for symbol in sorted(counts))


def layer_species(
    lattice: Sequence[Sequence[float]],
    positions_cartesian: Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    tolerance: float = LAYER_TOLERANCE,
) -> list[tuple[float, dict[str, int]]]:
    """Return ``(height, species counts)`` for each atomic layer, bottom upwards.

    Layers are the package-wide ones of ``core.layers.layer_partition``: atoms
    joined by steps of at most ``tolerance`` in height along the surface
    normal, reported at the mean height of the group.  Sharing that rule is
    what makes the two faces comparable -- it is the only grouping of the three
    the package used to carry that reads a slab the same way from either end,
    so ``symmetric_terminations`` no longer depends on which way round the
    structure was written.
    """

    heights = np.asarray(
        normal_heights(np.asarray(lattice, dtype=float), positions_cartesian), dtype=float
    )
    symbols = [str(value) for value in labels]
    if heights.shape[0] != len(symbols):
        raise ValueError("one species label per atom is required")
    if heights.size == 0:
        return []
    return [
        (
            height,
            {
                key: int(value)
                for key, value in sorted(Counter(symbols[index] for index in indices).items())
            },
        )
        for height, indices in layer_partition(heights, float(tolerance))
    ]


@dataclass(frozen=True)
class TerminationReport:
    """What a slab exposes and how its composition compares with the bulk."""

    bulk_formula: dict[str, int]
    slab_counts: dict[str, int]
    formula_units: float
    stoichiometric: bool
    excess: dict[str, int]
    bottom_termination: dict[str, int]
    top_termination: dict[str, int]
    symmetric_terminations: bool
    layer_count: int
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready description of the report."""

        return {
            "bulk_formula": dict(self.bulk_formula),
            "slab_counts": dict(self.slab_counts),
            "formula_units": float(self.formula_units),
            "stoichiometric": bool(self.stoichiometric),
            "excess": dict(self.excess),
            "bottom_termination": dict(self.bottom_termination),
            "top_termination": dict(self.top_termination),
            "symmetric_terminations": bool(self.symmetric_terminations),
            "layer_count": int(self.layer_count),
            "notes": list(self.notes),
        }


def termination_report(
    *,
    bulk_species: Sequence[str],
    bulk_counts: Sequence[int],
    slab_lattice: Sequence[Sequence[float]],
    slab_positions_cartesian: Sequence[Sequence[float]],
    slab_species: Sequence[str],
    slab_counts: Sequence[int],
    layer_tolerance: float = LAYER_TOLERANCE,
) -> TerminationReport:
    """Describe the composition and the two faces of a slab cut from a bulk cell.

    An elemental crystal always passes both tests, so its report carries no
    notes; the checks only ever have anything to say about a compound.
    """

    bulk_totals = {
        str(symbol): int(count) for symbol, count in zip(bulk_species, bulk_counts) if int(count) > 0
    }
    slab_totals = {
        str(symbol): int(count) for symbol, count in zip(slab_species, slab_counts) if int(count) > 0
    }
    formula = formula_unit(bulk_totals)

    # How many whole formula units the slab holds, and what is left over.  A
    # species the bulk does not contain cannot be part of a formula unit, so it
    # is reported as excess in full.
    if formula:
        units = min(
            (slab_totals.get(symbol, 0) // need for symbol, need in formula.items()),
            default=0,
        )
    else:
        units = 0
    excess = {
        symbol: int(slab_totals.get(symbol, 0) - units * formula.get(symbol, 0))
        for symbol in sorted(set(slab_totals) | set(formula))
    }
    excess = {symbol: value for symbol, value in excess.items() if value != 0}
    stoichiometric = not excess
    fractional_units = _fractional_units(slab_totals, formula)

    labels = expand_species(list(slab_species), [int(value) for value in slab_counts])
    layers = layer_species(
        slab_lattice, slab_positions_cartesian, labels, tolerance=float(layer_tolerance)
    )
    bottom = dict(layers[0][1]) if layers else {}
    top = dict(layers[-1][1]) if layers else {}
    symmetric = bool(layers) and bottom == top

    notes: list[str] = []
    if not stoichiometric:
        wording = ", ".join(
            f"{abs(value)} {'excess' if value > 0 else 'missing'} {symbol}"
            for symbol, value in excess.items()
        )
        notes.append(
            f"the slab holds {_format_counts(slab_totals)}, which is not a whole number of "
            f"{_format_counts(formula)} formula units ({wording}); a total energy of this cell "
            f"only means something against a chemical potential for the excess species, so cut "
            f"a stoichiometric thickness if a surface energy is wanted"
        )
    if layers and not symmetric:
        notes.append(
            f"the two faces of the slab are different terminations "
            f"({_format_counts(bottom)} below, {_format_counts(top)} above), so the cell carries "
            f"a dipole along the surface normal; use a dipole correction (IDIPOL/LDIPOL) or build "
            f"a symmetric slab before reading an energy off it"
        )

    return TerminationReport(
        bulk_formula=formula,
        slab_counts=slab_totals,
        formula_units=fractional_units,
        stoichiometric=stoichiometric,
        excess=excess,
        bottom_termination=bottom,
        top_termination=top,
        symmetric_terminations=symmetric,
        layer_count=len(layers),
        notes=notes,
    )


def _fractional_units(slab_totals: Mapping[str, int], formula: Mapping[str, int]) -> float:
    """Return how many formula units the slab holds, the limiting species deciding.

    The count is whole exactly when the slab is stoichiometric; ``2.5`` says the
    cell is half a formula unit short of three.  An empty formula -- a structure
    with no atoms -- has no meaningful count.
    """

    if not formula:
        return 0.0
    return float(min(float(slab_totals.get(symbol, 0)) / float(need) for symbol, need in formula.items()))
