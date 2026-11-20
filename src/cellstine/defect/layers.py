"""Choosing which atomic planes a defect is made in.

A defect analysis reports one representative of every orbit of symmetry-
equivalent sites, which is what a converged study of *distinct* defects needs.
It is not what a study of a slab needs: the two surfaces of a symmetric slab
are one orbit, and a mirror plane in the middle of a stack ties layer 1 to
layer N, so a single representative hides every layer but one.

This module turns an orbit back into one site *per atomic plane*.  For each
selected plane it keeps the members of the orbit that lie in it, so the site
that comes out is a genuine site of the structure and the multiplicity that
comes with it counts only the members of that plane.  Sites in different
planes of one orbit are equivalent by symmetry -- they are the same defect
seen at a different depth -- and that is exactly the point: the relaxation of a
vacancy at a surface and of the same vacancy in the interior are different
calculations.

Which planes are available, and how they are numbered, is fixed by the
direction of observation (see :mod:`cellstine.core.directions`): plane 1 is the
lowest along that direction and plane N the highest, so reversing the direction
reverses the numbering without changing which atoms share a plane.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping, Sequence

from .records import DefectSite

__all__ = [
    "LAYER_SELECTION_HELP",
    "layer_ids_of",
    "member_layer_ids",
    "parse_layer_selection",
    "resolve_layer_ids",
    "sites_by_layer",
    "sites_by_member",
]


from ..core.constants import LAYER_SELECTION_HELP  # re-exported for callers

_KEYWORDS = {"all", "top", "bottom", "surface", "surfaces", "interior", "inner", "middle", "centre", "center"}


def _selection_parts(spec: str | Sequence[int] | None) -> tuple[str, tuple[range | tuple[int, ...], ...]]:
    """Parse all tokens without expanding ranges before resolver bounds checks."""

    if spec is None:
        return "none", ()
    if not isinstance(spec, str):
        values = [int(value) for value in spec]
        if not values:
            return "none", ()
        return "explicit", (tuple(values),)
    text = spec.strip().lower()
    if not text:
        return "none", ()
    if text in _KEYWORDS:
        if text in {"surfaces"}:
            return "surface", ()
        if text in {"inner"}:
            return "interior", ()
        if text in {"centre", "center"}:
            return "middle", ()
        return text, ()
    parts: list[range | tuple[int, ...]] = []
    for chunk in text.replace(";", ",").split(","):
        token = chunk.strip()
        if not token:
            continue
        body = token.replace("..", "-")
        # A range needs a separating '-' that is not the sign of its first index.
        separator = body.find("-", 1)
        if separator > 0 and body[separator - 1].isdigit():
            start = int(body[:separator])
            stop = int(body[separator + 1 :])
            if start > stop:
                start, stop = stop, start
            parts.append(range(start, stop + 1))
        else:
            parts.append((int(body),))
    if not parts:
        raise ValueError(f"cannot read '{spec}' as a plane selection")
    return "explicit", tuple(parts)


def parse_layer_selection(spec: str | Sequence[int] | None) -> tuple[str, tuple[int, ...]]:
    """Return a keyword and full index tuple, including inclusive signed ranges."""

    keyword, parts = _selection_parts(spec)
    values: list[int] = []
    for part in parts:
        values.extend(part)
    return keyword, tuple(values)


def resolve_layer_ids(
    layers: Sequence[Mapping[str, Any]], spec: str | Sequence[int] | None
) -> tuple[int, ...] | None:
    """Return the plane ids a selection names, or ``None`` for no selection.

    Planes are numbered from 1 at the bottom of the structure along the
    direction of observation.  An index that no plane carries is an error, so a
    typed-in plane number is never silently ignored.

    Range bounds are checked before expansion; unavailable requests do not
    allocate one index per requested plane.
    """

    keyword, parts = _selection_parts(spec)
    if keyword == "none":
        return None
    available = [int(layer["layer_id"]) for layer in layers]
    if not available:
        raise ValueError("the structure has no atomic planes to choose from")
    ordered = sorted(available)
    if keyword == "all":
        return tuple(ordered)
    if keyword == "top":
        return (ordered[-1],)
    if keyword == "bottom":
        return (ordered[0],)
    if keyword == "surface":
        return (ordered[0],) if len(ordered) == 1 else (ordered[0], ordered[-1])
    if keyword == "interior":
        inner = ordered[1:-1]
        if not inner:
            raise ValueError(
                f"the structure has only {len(ordered)} atomic plane(s), so it has no interior"
            )
        return tuple(inner)
    if keyword == "middle":
        count = len(ordered)
        if count % 2 == 1:
            return (ordered[count // 2],)
        return (ordered[count // 2 - 1], ordered[count // 2])

    chosen: set[int] = set()
    for part in parts:
        if isinstance(part, range):
            start, stop = part.start, part.stop - 1
            # First invalid ordinal in the original ascending enumeration.
            bad = (start if start < -len(ordered) or start > len(ordered) else
                   0 if start <= 0 <= stop else
                   len(ordered) + 1 if stop > len(ordered) else None)
            if bad == 0:
                raise ValueError("atomic planes are numbered from 1; plane 0 does not exist")
            if bad is not None:
                raise ValueError(
                    f"plane {bad} does not exist; the structure has {len(ordered)} atomic plane(s)"
                )
        for value in part:
            index = int(value)
            if index == 0:
                raise ValueError("atomic planes are numbered from 1; plane 0 does not exist")
            if index > len(ordered) or index < -len(ordered):
                raise ValueError(
                    f"plane {index} does not exist; the structure has {len(ordered)} atomic plane(s)"
                )
            resolved = ordered[index - 1] if index > 0 else ordered[index]
            chosen.add(int(resolved))
    return tuple(sorted(chosen))


def member_layer_ids(site: DefectSite) -> tuple[int, ...]:
    """Every atomic plane in which the orbit of ``site`` has a member."""

    found: set[int] = set()
    for member in site.members:
        for layer_id in member.get("layer_ids", []) or []:
            if layer_id is not None:
                found.add(int(layer_id))
    if not found and site.layer_id is not None:
        found.add(int(site.layer_id))
    return tuple(sorted(found))


def layer_ids_of(sites: Iterable[DefectSite]) -> dict[str, tuple[int, ...]]:
    """Map each site id to the planes its orbit visits."""

    return {site.site_id: member_layer_ids(site) for site in sites}


def _member_matches(member: Mapping[str, Any], layer_id: int) -> bool:
    return int(layer_id) in {int(value) for value in (member.get("layer_ids") or []) if value is not None}


def _site_for_layer(site: DefectSite, layer_id: int, members: Sequence[Mapping[str, Any]]) -> DefectSite:
    representative = members[0]
    indices = [int(value) for value in (representative.get("indices") or [])]
    direct = tuple(float(value) for value in representative["direct"])
    cartesian = tuple(float(value) for value in representative["cartesian"])
    equivalent = sorted({int(value) for member in members for value in (member.get("indices") or [])})
    return replace(
        site,
        site_id=f"{site.site_id}_L{int(layer_id):02d}",
        layer_id=int(layer_id),
        direct=direct,
        cartesian=cartesian,
        equivalent_indices=equivalent if site.site_kind == "atom" else list(site.equivalent_indices),
        multiplicity=len(members),
        representative_index=indices[0] if indices else site.representative_index,
        pair_indices=indices[:2] if site.site_kind == "divacancy" and len(indices) >= 2 else list(site.pair_indices),
        members=[dict(member) for member in members],
    )


def _members_for_expansion(site: DefectSite, *, require_layers: bool = False) -> Sequence[Mapping[str, Any]]:
    """Restore a known singleton, never guess an unavailable multi-member orbit."""

    if site.members or site.site_kind == "adatom":
        return site.members
    if int(site.multiplicity) > 1 or (
        site.site_kind == "atom" and len({int(index) for index in site.equivalent_indices}) > 1
    ) or (
        require_layers and (site.site_kind == "divacancy" or site.layer_id is None)
    ):
        raise ValueError(f"{site.site_id} has no orbit members. Reanalyse the structure before expanding its sites.")
    indices = (list(site.pair_indices[:2]) if site.site_kind == "divacancy" else
               ([site.representative_index] if site.representative_index is not None else list(site.equivalent_indices))
               if site.site_kind == "atom" else [])
    return [{"indices": indices, "direct": list(site.direct), "cartesian": list(site.cartesian),
             "layer_ids": [] if site.layer_id is None else [int(site.layer_id)]}]


def sites_by_layer(
    sites: Sequence[DefectSite], layer_ids: Sequence[int]
) -> tuple[list[DefectSite], list[str]]:
    """Split each orbit into one site per selected atomic plane.

    Sites whose orbit has no member in any selected plane are dropped.  Sites
    that do not belong to a plane at all -- an adatom sits above the surface,
    not in it -- are kept untouched, with a note saying so, because the plane
    they would be filed under is not a property of the defect.

    Older records with a known single representative remain usable. Expanding
    an unavailable multi-member orbit requires a new analysis of the structure.
    """

    wanted = [int(value) for value in layer_ids]
    expanded: list[DefectSite] = []
    notes: list[str] = []
    unplaced = 0
    for site in sites:
        if not wanted:
            if not member_layer_ids(site):
                unplaced += 1
                expanded.append(site)
            continue
        by_layer: dict[int, list[Mapping[str, Any]]] = {}
        for member in _members_for_expansion(site, require_layers=True):
            for layer_id in {int(value) for value in (member.get("layer_ids") or []) if value is not None}:
                by_layer.setdefault(layer_id, []).append(member)
        if not by_layer and site.layer_id is None:
            unplaced += 1
            expanded.append(site)
            continue
        for layer_id in wanted:
            members = by_layer.get(layer_id, [])
            if not members:
                continue
            expanded.append(_site_for_layer(site, layer_id, members))
    if unplaced:
        notes.append(
            f"{unplaced} site(s) do not lie in an atomic plane (adatoms sit above the surface); "
            "the plane selection does not apply to them."
        )
    return expanded, notes


def _site_for_member(site: DefectSite, member: Mapping[str, Any]) -> DefectSite:
    """One site standing for a single member of an orbit."""

    indices = [int(value) for value in (member.get("indices") or [])]
    layer_ids = [int(value) for value in (member.get("layer_ids") or []) if value is not None]
    token = "-".join(f"{index:03d}" for index in indices) if indices else "x"
    return replace(
        site,
        site_id=f"{site.site_id}_M{token}",
        layer_id=layer_ids[0] if layer_ids else site.layer_id,
        direct=tuple(float(value) for value in member["direct"]),
        cartesian=tuple(float(value) for value in member["cartesian"]),
        equivalent_indices=list(indices) if site.site_kind == "atom" else list(site.equivalent_indices),
        multiplicity=1,
        representative_index=indices[0] if indices else site.representative_index,
        pair_indices=indices[:2] if site.site_kind == "divacancy" and len(indices) >= 2 else list(site.pair_indices),
        members=[dict(member)],
    )


def sites_by_member(
    sites: Sequence[DefectSite], layer_ids: Sequence[int] | None = None
) -> tuple[list[DefectSite], list[str]]:
    """Expand each orbit into one site per member, not one per orbit.

    This is what ``generate='all'`` asks for: every atom of the host that the
    defect could be made at, rather than one representative of each orbit of
    symmetry-equivalent atoms.  The structures it writes are related to one
    another by a symmetry of the perfect host, so their energies agree; what
    they are for is a study that breaks that symmetry later, such as a pair of
    defects or an applied field.  When ``layer_ids`` is given only members in
    those atomic planes are kept. Adatoms with no host-plane orbit are kept
    with a note. A known singleton in an older record is restored; unknown
    multi-member orbit data requires reanalysis rather than partial output.
    """

    wanted = None if layer_ids is None else {int(value) for value in layer_ids}
    expanded: list[DefectSite] = []
    notes: list[str] = []
    unplaced = 0
    retained_singletons = 0
    expanded_members = False
    for site in sites:
        members = _members_for_expansion(site, require_layers=wanted is not None)
        if not members:
            unplaced += 1
            expanded.append(site)
            continue
        if not site.members:
            member_layers = {int(value) for value in members[0]["layer_ids"]}
            if wanted is None or member_layers & wanted:
                expanded.append(site)
                retained_singletons += 1
            continue
        for member in members:
            if wanted is not None:
                member_layers = {int(value) for value in (member.get("layer_ids") or []) if value is not None}
                if not (member_layers & wanted):
                    continue
            expanded.append(_site_for_member(site, member))
            expanded_members = True
    if unplaced:
        notes.append(
            f"{unplaced} site(s) have no orbit of equivalent atoms to expand and were kept as they are."
        )
    if retained_singletons:
        notes.append(
            f"{retained_singletons} site(s) have only a known single representative and were kept without inventing an orbit."
        )
    if expanded_members:
        notes.append(
            "Every member of each orbit was expanded, so equivalent copies of the same defect "
            "are written as separate structures."
        )
    return expanded, notes
