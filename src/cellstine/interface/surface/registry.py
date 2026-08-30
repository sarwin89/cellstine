"""Enumeration of the distinct ways two close-packed slabs can be put in contact.

The letters ``A``, ``B`` and ``C`` that name the layers of a close-packed slab
carry two gauge freedoms, so several labelled combinations describe one and the
same structure.  This module writes an interface as a word of layer-to-layer
steps in ``Z/3``, quotients that word by the operations which only relabel it,
and reports one option per class.  The counting is proved in the external
Lean reference
``aristotle-lean-reference/RequestProject/StackingRegistry.lean`` and the module docstring of
``stacking.py`` explains the two gauges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .stacking import COSET_LETTERS, POSITION_TOLERANCE, StackingAnalysis, sense_label

__all__ = [
    "RegistryOption",
    "canonical_configuration",
    "configuration_orbit",
    "configuration_word",
    "enumerate_registry_options",
    "format_registry_table",
    "mirror_configuration",
    "registry_contact_label",
    "registry_kind",
    "select_registry_option",
    "slabs_are_interchangeable",
]


def mirror_configuration(
    bottom_increments: Sequence[int], top_increments: Sequence[int], delta: int
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    """Reflect a labelled interface configuration.

    A reflection negates every coset label, hence every layer increment and the
    contact difference.
    """

    return (
        tuple((-int(value)) % 3 for value in bottom_increments),
        tuple((-int(value)) % 3 for value in top_increments),
        (-int(delta)) % 3,
    )


def configuration_word(
    bottom_increments: Sequence[int], top_increments: Sequence[int], delta: int
) -> tuple[int, ...]:
    """Return the interface as one word of layer-to-layer coset steps.

    The whole stack, both slabs and the contact between them, is a single list
    of steps in ``Z/3``; the contact step sits between the two slabs.
    """

    return (
        tuple(int(value) % 3 for value in bottom_increments)
        + (int(delta) % 3,)
        + tuple(int(value) % 3 for value in top_increments)
    )


def configuration_orbit(
    word: Sequence[int], *, slabs_interchangeable: bool = False
) -> list[tuple[int, ...]]:
    """Return the words of every structure congruent to ``word``.

    Reflecting the interface in a vertical plane negates every step, and
    turning it over — swapping which slab is below — reverses the word and
    negates it.  Turning it over is only a relabelling of the same physical
    structure when the two slabs can trade places, which is what
    ``slabs_interchangeable`` records.
    """

    straight = tuple(int(value) % 3 for value in word)
    mirrored = tuple((-value) % 3 for value in straight)
    orbit = [straight, mirrored]
    if slabs_interchangeable:
        orbit.append(tuple(reversed(mirrored)))
        orbit.append(tuple(reversed(straight)))
    return orbit


def canonical_configuration(
    bottom_increments: Sequence[int],
    top_increments: Sequence[int],
    delta: int,
    *,
    slabs_interchangeable: bool = False,
) -> tuple[int, ...]:
    """Return the canonical word of a configuration's congruence class.

    Two configurations describe structures related by an isometry exactly when
    they share a canonical form, so this is the key that removes the duplicate
    options.
    """

    word = configuration_word(bottom_increments, top_increments, delta)
    return min(configuration_orbit(word, slabs_interchangeable=slabs_interchangeable))


def registry_contact_label(bottom_label: str, delta: int) -> str:
    """Name the contact, e.g. ``C-A`` for a ``C``-terminated slab and ``delta=1``."""

    if bottom_label not in COSET_LETTERS:
        return "?-?"
    bottom_index = COSET_LETTERS.index(bottom_label)
    return f"{bottom_label}-{COSET_LETTERS[(bottom_index + int(delta)) % 3]}"


def registry_kind(delta: int, bottom_sense: int, *, bottom_last_step: int | None = None) -> str:
    """Classify a contact as eclipsed, continuing, or faulted.

    What decides between the two hollows is the step the bottom slab takes into
    its own outermost layer, because that is the direction its sequence would
    continue in.  For a uniform ``ABC`` slab that step is the stacking sense,
    but a slab such as ``ABABAB`` has no sense and still has a well defined
    outermost step, so ``bottom_last_step`` is preferred when it is given.
    """

    value = int(delta) % 3
    if value == 0:
        return "eclipsed"
    step = int(bottom_sense if bottom_last_step is None else bottom_last_step) % 3
    if step == 0:
        return f"hollow_{value}"
    return "fcc_hollow" if value == step else "hcp_hollow"


REGISTRY_DESCRIPTIONS = {
    "eclipsed": "the two contacting layers sit directly on top of each other (AA)",
    "fcc_hollow": "the upper layer continues the sequence of the lower slab (fcc hollow)",
    "hcp_hollow": "the upper layer sits above the second layer of the lower slab (hcp hollow)",
}


@dataclass(frozen=True)
class RegistryOption:
    """One distinct way of putting two close-packed slabs in contact."""

    index: int
    bottom_mirrored: bool
    top_mirrored: bool
    bottom_sequence: str
    top_sequence: str
    bottom_sense: int
    top_sense: int
    delta: int
    contact: str
    kind: str
    description: str
    equivalent_to: int | None = None
    relation: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "index": int(self.index),
            "bottom_mirrored": bool(self.bottom_mirrored),
            "top_mirrored": bool(self.top_mirrored),
            "bottom_sequence": str(self.bottom_sequence),
            "top_sequence": str(self.top_sequence),
            "bottom_sense": sense_label(self.bottom_sense),
            "top_sense": sense_label(self.top_sense),
            "delta": int(self.delta),
            "contact": str(self.contact),
            "kind": str(self.kind),
            "description": str(self.description),
            "equivalent_to": None if self.equivalent_to is None else int(self.equivalent_to),
            "relation": str(self.relation),
        }


def _mirrored_sequence(sequence: str) -> str:
    letters = []
    for index, letter in enumerate(sequence):
        if letter not in COSET_LETTERS:
            return sequence
        if index == 0:
            letters.append(letter)
            continue
        step = (COSET_LETTERS.index(sequence[index]) - COSET_LETTERS.index(sequence[index - 1])) % 3
        previous = COSET_LETTERS.index(letters[-1])
        letters.append(COSET_LETTERS[(previous - step) % 3])
    return "".join(letters)


def enumerate_registry_options(
    bottom: StackingAnalysis,
    top: StackingAnalysis,
    *,
    include_equivalent: bool = False,
    slabs_interchangeable: bool = False,
) -> list[RegistryOption]:
    """List the distinct contacts between two close-packed slabs.

    ``top`` must be analysed in the gauge of ``bottom``, i.e. with
    ``hollow_cartesian=bottom.hollow_cartesian``.  Both stacking senses of the
    top slab are crossed with each of the three contacts, and the options that
    only differ by the labelling gauge are dropped: ``A-A``, ``B-B`` and
    ``C-C`` are one option because the origin is arbitrary, and reversing both
    slabs at once only reflects the whole interface.  Two ``ABC``-type slabs
    therefore give six entries out of the twelve labelled combinations.

    ``slabs_interchangeable`` says that the two slabs can trade places, which
    is the case when they are the same slab.  Turning such an interface over is
    then a relabelling too, and it merges further options -- two equally thick
    ``ABC`` slabs of one material have five distinct interfaces, not six,
    because the two twinned contacts are one boundary seen from either side.

    With ``include_equivalent`` the mirrored bottom slab is enumerated too, so
    every labelled combination appears and the removed ones are marked with the
    option they are congruent to.
    """

    if not bottom.close_packed:
        raise ValueError(f"the bottom slab is not close packed: {bottom.reason}")
    if not top.close_packed:
        raise ValueError(f"the top slab is not close packed: {top.reason}")

    bottom_choices = [(False, bottom.increments, bottom.sequence, bottom.sense)]
    if include_equivalent and bottom.reversible:
        bottom_choices.append(
            (
                True,
                mirror_configuration(bottom.increments, (), 0)[0],
                _mirrored_sequence(bottom.sequence),
                -bottom.sense,
            )
        )
    top_choices = [(False, top.increments, top.sequence, top.sense)]
    if top.reversible:
        top_choices.append(
            (
                True,
                mirror_configuration(top.increments, (), 0)[0],
                _mirrored_sequence(top.sequence),
                -top.sense,
            )
        )

    options: list[RegistryOption] = []
    seen: dict[tuple[int, ...], int] = {}
    for bottom_mirrored, bottom_increments, bottom_sequence, bottom_sense in bottom_choices:
        for top_mirrored, top_increments, top_sequence, top_sense in top_choices:
            for delta in range(3):
                word = configuration_word(bottom_increments, top_increments, delta)
                key = min(
                    configuration_orbit(word, slabs_interchangeable=slabs_interchangeable)
                )
                duplicate = seen.get(key)
                if duplicate is not None and not include_equivalent:
                    continue
                index = len(options) + 1
                relation = ""
                if duplicate is not None:
                    mirrored_word = tuple((-value) % 3 for value in word)
                    earlier = options[duplicate - 1]
                    earlier_word = configuration_word(
                        (
                            tuple((-value) % 3 for value in bottom.increments)
                            if earlier.bottom_mirrored
                            else bottom.increments
                        ),
                        (
                            tuple((-value) % 3 for value in top.increments)
                            if earlier.top_mirrored
                            else top.increments
                        ),
                        earlier.delta,
                    )
                    relation = (
                        "mirror image" if mirrored_word == earlier_word else "same interface turned over"
                    )
                else:
                    seen[key] = index
                kind = registry_kind(
                    delta,
                    bottom_sense,
                    bottom_last_step=bottom_increments[-1] if bottom_increments else None,
                )
                bottom_label = bottom_sequence[-1] if bottom_sequence else "?"
                options.append(
                    RegistryOption(
                        index=index,
                        bottom_mirrored=bottom_mirrored,
                        top_mirrored=top_mirrored,
                        bottom_sequence=bottom_sequence,
                        top_sequence=top_sequence,
                        bottom_sense=bottom_sense,
                        top_sense=top_sense,
                        delta=int(delta),
                        contact=registry_contact_label(bottom_label, delta),
                        kind=kind,
                        description=REGISTRY_DESCRIPTIONS.get(kind, "distinct hollow contact"),
                        equivalent_to=duplicate,
                        relation=relation,
                    )
                )
    return options


def slabs_are_interchangeable(
    bottom,
    top,
    bottom_analysis: StackingAnalysis,
    top_analysis: StackingAnalysis,
    *,
    tolerance: float = POSITION_TOLERANCE,
) -> bool:
    """True when turning the interface over gives back the same pair of slabs.

    The test compares the two slabs layer by layer outwards from the contact:
    same number of layers, same species in each, and the same layer spacings.
    Under those conditions the flipped bottom slab and the top slab differ only
    by an isometry, so the whole interface may be turned over.
    """

    from ...core.species import expand_species

    if len(bottom_analysis.layers) != len(top_analysis.layers):
        return False

    def profile(record, analysis, reverse: bool):
        symbols = expand_species(record.species, record.counts)
        layers = list(analysis.layers[::-1] if reverse else analysis.layers)
        composition = [
            tuple(sorted(symbols[index] for index in layer.atom_indices)) for layer in layers
        ]
        heights = [float(layer.height) for layer in layers]
        spacings = [abs(heights[index + 1] - heights[index]) for index in range(len(heights) - 1)]
        return composition, spacings

    bottom_composition, bottom_spacings = profile(bottom, bottom_analysis, True)
    top_composition, top_spacings = profile(top, top_analysis, False)
    if bottom_composition != top_composition:
        return False
    return all(
        abs(first - second) <= tolerance for first, second in zip(bottom_spacings, top_spacings)
    )


def select_registry_option(
    options: Sequence[RegistryOption], selector: str | int | None
) -> RegistryOption | None:
    """Pick one enumerated option by index, contact label, or kind."""

    if selector is None:
        return None
    if isinstance(selector, bool):
        raise ValueError("registry must be an index, a contact such as A-C, or a kind such as fcc")
    if isinstance(selector, int):
        for option in options:
            if option.index == int(selector):
                return option
        raise ValueError(f"registry index {selector} is not one of the {len(options)} distinct options")
    text = str(selector).strip().lower().replace("_", "-")
    if text.isdigit():
        return select_registry_option(options, int(text))
    aliases = {
        "aa": "eclipsed",
        "top": "eclipsed",
        "eclipsed": "eclipsed",
        "fcc": "fcc_hollow",
        "fcc-hollow": "fcc_hollow",
        "continue": "fcc_hollow",
        "hcp": "hcp_hollow",
        "hcp-hollow": "hcp_hollow",
        "fault": "hcp_hollow",
    }
    if text in aliases:
        for option in options:
            if option.kind == aliases[text]:
                return option
        raise ValueError(f"no distinct option of kind {aliases[text]!r} is available")
    contact = text.upper().replace("-", "")
    if len(contact) == 2 and all(letter in COSET_LETTERS for letter in contact):
        # Only the difference of the two letters is physical, so A-A, B-B and
        # C-C name the same contact, and so do C-A, A-B and B-C.
        wanted = (COSET_LETTERS.index(contact[1]) - COSET_LETTERS.index(contact[0])) % 3
        for option in options:
            if int(option.delta) % 3 == wanted:
                return option
        raise ValueError(
            f"contact {contact[0]}-{contact[1]} is not one of the distinct options; "
            + ", ".join(option.contact for option in options)
        )
    raise ValueError(
        f"unknown registry {selector!r}; use an index, a contact such as A-C, or fcc/hcp/eclipsed"
    )


def format_registry_table(options: Iterable[RegistryOption]) -> str:
    """Render the distinct interface options as a plain table."""

    rows = [
        (
            "idx",
            "bottom",
            "top",
            "contact",
            "kind",
            "note",
        )
    ]
    for option in options:
        rows.append(
            (
                str(option.index),
                f"{option.bottom_sequence}{' (mirrored)' if option.bottom_mirrored else ''}",
                f"{option.top_sequence}{' (mirrored)' if option.top_mirrored else ''}",
                option.contact,
                option.kind,
                "" if option.equivalent_to is None else f"{option.relation} of {option.equivalent_to}",
            )
        )
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    lines = []
    for position, row in enumerate(rows):
        lines.append("  ".join(value.ljust(widths[column]) for column, value in enumerate(row)).rstrip())
        if position == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)
