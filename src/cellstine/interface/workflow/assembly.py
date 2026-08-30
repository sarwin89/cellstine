"""Slab assembly helpers shared by the interface build paths.

Everything here works on structure records rather than on the workflow: reading
Miller notation, describing the contact two close-packed slabs make, applying
the requested stacking order and registry, and stacking two slabs into one cell
with an exact interface gap and vacuum.  :mod:`cellstine.interface.workflow.interface`
orchestrates them; keeping them apart keeps that module readable and lets the
CLI import the parser without pulling the whole workflow in.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ...core.contacts import layer_contact_report
from ...core.naming import safe_token
from ...core.species import expand_species
from ...io import native as io_mod
from ..surface import registry as registry_mod
from ..surface import stacking as stacking_mod

__all__ = [
    "parse_miller_notation",
    "safe_token",
    "group_by_species",
    "contact_summary",
    "realised_stacking_summary",
    "prepare_stacking",
    "reported_kind",
    "interface_contact_report",
    "slab_vacuum_thickness",
    "stack_structures",
    "stacked_cartesian",
]


def parse_miller_notation(miller: str | Sequence[int]) -> tuple[int, int, int]:
    if not isinstance(miller, str):
        values = [int(value) for value in miller]
        if len(values) != 3:
            raise ValueError("Miller indices must have exactly three values")
        return values[0], values[1], values[2]
    raw = miller.strip()
    if "," not in raw and ";" not in raw and " " not in raw:
        tokens = []
        index = 0
        while index < len(raw):
            char = raw[index]
            if not char.isdigit():
                raise ValueError("compact Miller notation must look like 111, 001, 1x11, or 111x")
            token = char
            if index + 1 < len(raw) and raw[index + 1].lower() == "x":
                token += "x"
                index += 1
            tokens.append(token)
            index += 1
    else:
        tokens = [token.strip() for token in miller.replace(";", ",").replace(" ", ",").split(",") if token.strip()]
    if len(tokens) != 3:
        raise ValueError("Miller indices must be given as h,k,l such as 1,1,1 or compact notation such as 111, 001, or 111x")
    values = []
    for token in tokens:
        if token.lower().endswith("x"):
            values.append(-int(token[:-1]))
        else:
            values.append(int(token))
    if values == [0, 0, 0]:
        raise ValueError("Miller indices cannot all be zero")
    return int(values[0]), int(values[1]), int(values[2])


def group_by_species(record, positions_direct: np.ndarray, selective_flags):
    expanded_species = expand_species(record.species, record.counts)
    grouped_positions: dict[str, list[np.ndarray]] = {}
    grouped_flags: dict[str, list[tuple[str, str, str] | None]] = {}
    species_order: list[str] = []
    for index, symbol in enumerate(expanded_species):
        if symbol not in grouped_positions:
            grouped_positions[symbol] = []
            grouped_flags[symbol] = []
            species_order.append(symbol)
        grouped_positions[symbol].append(np.array(positions_direct[index], dtype=float))
        if selective_flags is None:
            grouped_flags[symbol].append(None)
        else:
            grouped_flags[symbol].append(tuple(selective_flags[index]))
    ordered_positions = []
    ordered_flags: list[tuple[str, str, str]] = []
    counts = []
    for symbol in species_order:
        ordered_positions.extend(grouped_positions[symbol])
        counts.append(len(grouped_positions[symbol]))
        if selective_flags is not None:
            ordered_flags.extend(flag for flag in grouped_flags[symbol] if flag is not None)
    flags_out = ordered_flags if selective_flags is not None else None
    return np.array(ordered_positions, dtype=float), counts, species_order, flags_out


def contact_summary(
    bottom_analysis,
    top_analysis,
    *,
    delta: int | None,
    bottom_mirrored: bool,
    top_mirrored: bool,
) -> dict:
    """Describe the contact two analysed slabs make.

    ``delta`` is the contact that was *asked for*; when nothing was asked for,
    the contact the two slabs happen to make is read off their labels instead,
    so a built interface always reports which of the three contacts it is.
    """

    requested = delta is not None
    if delta is None:
        bottom_coset = bottom_analysis.top_coset
        top_coset = top_analysis.bottom_coset
        if bottom_coset is None or top_coset is None:
            return {}
        delta = (int(top_coset) - int(bottom_coset)) % 3
    delta = int(delta) % 3
    last_step = bottom_analysis.increments[-1] if bottom_analysis.increments else None
    kind = registry_mod.registry_kind(delta, bottom_analysis.sense, bottom_last_step=last_step)
    summary = {
        "bottom_sequence": bottom_analysis.sequence,
        "top_sequence": top_analysis.sequence,
        "bottom_sense": bottom_analysis.sense_label,
        "top_sense": top_analysis.sense_label,
        "bottom_mirrored": bool(bottom_mirrored),
        "top_mirrored": bool(top_mirrored),
        "delta": delta,
        "contact": registry_mod.registry_contact_label(bottom_analysis.top_label, delta),
        "kind": kind,
        "registry_requested": bool(requested),
    }
    if not requested and kind == "eclipsed":
        summary["note"] = (
            "the slabs meet eclipsed, one layer directly above the other; "
            "pass --registry fcc for the contact that continues the bulk stacking"
        )
    return summary


def realised_stacking_summary(bottom, top) -> dict:
    """Report the contact of two slabs that are stacked exactly as they arrive."""

    bottom_analysis = stacking_mod.analyse_stacking(bottom)
    if not bottom_analysis.close_packed:
        raise ValueError(bottom_analysis.reason)
    top_analysis = stacking_mod.analyse_stacking(
        top, hollow_cartesian=bottom_analysis.hollow_cartesian
    )
    if not top_analysis.close_packed:
        raise ValueError(top_analysis.reason)
    return contact_summary(
        bottom_analysis,
        top_analysis,
        delta=None,
        bottom_mirrored=False,
        top_mirrored=False,
    )


def prepare_stacking(
    bottom,
    top,
    *,
    bottom_stacking: str,
    top_stacking: str,
    registry: str | int | None,
    include_equivalent: bool = False,
):
    """Apply the requested stacking senses and contact to a pair of slabs.

    ``bottom`` and ``top`` must already share an in-plane cell.  The bottom slab
    fixes the ``A -> B -> C`` gauge, the top slab is described in it, and the
    contact is set exactly by an in-plane translation of the top slab.  Nothing
    happens, and no close-packed analysis is demanded, when the caller asks for
    neither a sense nor a contact.
    """

    bottom_choice = stacking_mod.normalise_stacking_choice(bottom_stacking)
    top_choice = stacking_mod.normalise_stacking_choice(top_stacking)
    if bottom_choice in {"abc", "cba"}:
        raise ValueError(
            "a slab has no handedness of its own; use keep or mirror for the bottom slab and "
            "abc or cba for the top slab, which is described relative to the bottom one"
        )
    # Nothing was asked for, so nothing is moved -- but the two slabs still meet
    # in a definite contact, and reporting which one it is costs one analysis.
    passive = bottom_choice == "keep" and top_choice == "keep" and registry is None
    if passive:
        try:
            return bottom, top, realised_stacking_summary(bottom, top)
        except ValueError:
            return bottom, top, None

    incoming = stacking_mod.analyse_stacking(bottom)
    if not incoming.close_packed:
        raise ValueError(f"the bottom slab is not close packed: {incoming.reason}")
    # The bottom slab as it arrives fixes the A -> B -> C direction; every
    # analysis below is read in that one gauge, so the letters keep meaning the
    # same thing even after a slab has been mirrored.
    gauge = incoming.hollow_cartesian
    top_incoming = stacking_mod.analyse_stacking(top, hollow_cartesian=gauge)
    if not top_incoming.close_packed:
        raise ValueError(f"the top slab is not close packed: {top_incoming.reason}")

    interchangeable = registry_mod.slabs_are_interchangeable(
        bottom, top, incoming, top_incoming
    )
    options = registry_mod.enumerate_registry_options(
        incoming,
        top_incoming,
        include_equivalent=include_equivalent,
        slabs_interchangeable=interchangeable,
    )

    mirror_bottom = bottom_choice == "mirror"
    if top_choice == "mirror":
        mirror_top = True
    elif top_choice in {"abc", "cba"}:
        if top_incoming.sense == 0:
            raise ValueError(
                f"the top slab stacks as {top_incoming.sequence!r}, which is neither ABC nor CBA, "
                "so its stacking sense cannot be set; use keep"
            )
        mirror_top = top_incoming.sense != (1 if top_choice == "abc" else -1)
    else:
        mirror_top = False

    # A contact or a kind names the option with the requested stacking senses,
    # and is looked up among every labelled combination: a contact that the
    # reported table drops as congruent to another one is still buildable once
    # the senses of both slabs are pinned down.  An index instead names a row of
    # the reported table and therefore carries its own senses.
    by_index = isinstance(registry, int) or (
        isinstance(registry, str) and registry.strip().isdigit()
    )
    selectable = options
    if not by_index:
        labelled = registry_mod.enumerate_registry_options(
            incoming,
            top_incoming,
            include_equivalent=True,
            slabs_interchangeable=interchangeable,
        )
        # A slab that reads the same either way up is never enumerated mirrored,
        # and reflecting it changes nothing, so it never constrains the choice.
        wanted_bottom = mirror_bottom and incoming.reversible
        wanted_top = mirror_top and top_incoming.reversible
        selectable = [
            entry
            for entry in labelled
            if bool(entry.top_mirrored) == wanted_top
            and bool(entry.bottom_mirrored) == wanted_bottom
        ]
    option = registry_mod.select_registry_option(selectable, registry)
    if option is not None:
        if top_choice != "keep" and bool(option.top_mirrored) != mirror_top:
            raise ValueError(
                f"registry option {option.index} and top_stacking={top_stacking!r} ask for "
                "different stacking senses of the top slab; select the option by contact instead"
            )
        if bottom_choice != "keep" and bool(option.bottom_mirrored) != mirror_bottom:
            raise ValueError(
                f"registry option {option.index} and bottom_stacking={bottom_stacking!r} ask for "
                "different stacking senses of the bottom slab"
            )
        mirror_bottom = mirror_bottom or bool(option.bottom_mirrored)
        mirror_top = mirror_top or bool(option.top_mirrored)

    if mirror_bottom:
        bottom = stacking_mod.mirror_structure(bottom)
    if mirror_top:
        top = stacking_mod.mirror_structure(top)
    bottom_analysis = stacking_mod.analyse_stacking(bottom, hollow_cartesian=gauge)
    top_analysis = stacking_mod.analyse_stacking(top, hollow_cartesian=gauge)

    delta = None if option is None else int(option.delta)
    if delta is not None:
        shift = stacking_mod.registry_shift_direct(
            bottom, top, bottom_analysis, top_analysis, delta, hollow=gauge
        )
        top = stacking_mod.shift_structure_inplane(top, shift)
        top_analysis = stacking_mod.analyse_stacking(top, hollow_cartesian=gauge)

    summary = contact_summary(
        bottom_analysis,
        top_analysis,
        delta=delta,
        bottom_mirrored=bool(mirror_bottom),
        top_mirrored=bool(mirror_top),
    )
    summary["distinct_options"] = sum(1 for entry in options if entry.equivalent_to is None)
    return bottom, top, summary


def reported_kind(meta: dict[str, object]) -> str:
    """Name the kind an input was read as, saying so when it was detected."""

    kind = str(meta.get("kind", "surface"))
    return f"{kind} (detected)" if meta.get("detected") else kind


def slab_vacuum_thickness(record) -> float:
    """Return the vacuum thickness of a slab, i.e. c minus the occupied span."""

    cartesian = np.asarray(record.positions_cartesian, dtype=float)
    c_length = abs(float(np.asarray(record.lattice, dtype=float)[2, 2]))
    if not cartesian.size:
        return c_length
    span = float(cartesian[:, 2].max() - cartesian[:, 2].min())
    return max(c_length - span, 0.0)


def stacked_cartesian(bottom, top, *, gap: float, vacuum: float):
    """Return the stacked cell and the Cartesian coordinates of the two slabs.

    This is the geometry of :func:`stack_structures` on its own, before the
    atoms are regrouped by species: the caller keeps hold of which atom belongs
    to which slab, which is what measuring the contact across the interface
    needs.
    """

    bottom_cartesian = np.array(bottom.positions_cartesian, dtype=float, copy=True)
    top_cartesian = np.array(top.positions_cartesian, dtype=float, copy=True)
    bottom_min = float(bottom_cartesian[:, 2].min()) if bottom_cartesian.size else 0.0
    bottom_max = float(bottom_cartesian[:, 2].max()) if bottom_cartesian.size else 0.0
    top_min = float(top_cartesian[:, 2].min()) if top_cartesian.size else 0.0
    top_max = float(top_cartesian[:, 2].max()) if top_cartesian.size else 0.0
    bottom_thickness = bottom_max - bottom_min
    top_thickness = top_max - top_min
    stack_thickness = bottom_thickness + float(gap) + top_thickness
    final_c_length = stack_thickness + float(vacuum)
    if final_c_length <= 1e-12:
        raise ValueError("interface c axis would have zero length")
    base = 0.5 * float(vacuum)
    bottom_cartesian[:, 2] += base - bottom_min
    top_cartesian[:, 2] += base + bottom_thickness + float(gap) - top_min
    final_lattice = np.array(bottom.lattice, dtype=float, copy=True)
    final_lattice[2] = np.array([0.0, 0.0, final_c_length], dtype=float)

    return final_lattice, bottom_cartesian, top_cartesian


def interface_contact_report(bottom, top, *, gap: float, vacuum: float) -> dict:
    """Measure the closest approach the two slabs of an interface make.

    The interface gap separates the two slabs along the surface normal; what
    says whether they are touching is the shortest distance between an atom of
    one slab and an atom of the other, taken over the periodic images so that a
    contact made around a cell face counts as well.
    """

    final_lattice, bottom_cartesian, top_cartesian = stacked_cartesian(
        bottom, top, gap=float(gap), vacuum=float(vacuum)
    )
    return layer_contact_report(
        lattice=final_lattice,
        first_cartesian=bottom_cartesian,
        second_cartesian=top_cartesian,
        first_species=expand_species(bottom.species, bottom.counts),
        second_species=expand_species(top.species, top.counts),
        subject="interface",
        requested=float(gap),
        requested_name="gap",
    )


def stack_structures(bottom, top, *, gap: float, vacuum: float):
    """Stack ``top`` above ``bottom`` with an exact interface gap and vacuum gap.

    The two slabs are separated by ``gap`` and the cell is sized so that the
    periodic image of the bottom slab sits exactly ``vacuum`` above the top of
    the upper slab.  The stack is centred, so both surfaces see the same amount
    of vacuum on either side of the boundary.
    """

    final_lattice, bottom_cartesian, top_cartesian = stacked_cartesian(
        bottom, top, gap=float(gap), vacuum=float(vacuum)
    )
    bottom_direct = io_mod.cartesian_to_direct(bottom_cartesian, final_lattice)
    top_direct = io_mod.cartesian_to_direct(top_cartesian, final_lattice)
    bottom_positions, bottom_counts, bottom_species, bottom_flags = group_by_species(bottom, bottom_direct, bottom.selective_flags)
    top_positions, top_counts, top_species, top_flags = group_by_species(top, top_direct, top.selective_flags)

    species = []
    counts = []
    grouped_flags = []
    grouped_positions = []
    all_species = bottom_species + [symbol for symbol in top_species if symbol not in bottom_species]
    for symbol in all_species:
        symbol_positions = []
        symbol_flags = []
        for current_species, current_counts, current_positions, current_flags in (
            (bottom_species, bottom_counts, bottom_positions, bottom_flags),
            (top_species, top_counts, top_positions, top_flags),
        ):
            offset = 0
            for species_name, count in zip(current_species, current_counts):
                block = current_positions[offset : offset + count]
                block_flags = None if current_flags is None else current_flags[offset : offset + count]
                if species_name == symbol:
                    symbol_positions.extend(np.array(item, dtype=float) for item in block)
                    if block_flags is not None:
                        symbol_flags.extend(tuple(item) for item in block_flags)
                offset += count
        if symbol_positions:
            species.append(symbol)
            counts.append(len(symbol_positions))
            grouped_positions.extend(symbol_positions)
            grouped_flags.extend(symbol_flags)
    flags_out = grouped_flags if grouped_flags else None
    return final_lattice, np.array(grouped_positions, dtype=float), counts, species, flags_out
