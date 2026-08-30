"""Native vectorised Gram-form search for commensurate bilayer supercells.

The engine enumerates reduced positive-definite Gram forms, folds them by the
point group of each layer, joins the two families through the Loewner
inequalities implied by the strain budgets, and finishes in closed form with
Lagrange--Gauss gauge reduction, metric vector shells, reduced-basis arc
generation and a four-probe bucketed Gram join.

``top_strain`` and ``bottom_strain`` are absolute bounds on principal logarithmic
(Hencky) strain.  A budget ``e`` therefore permits principal stretches in
``[exp(-e), exp(e)]``; it is not an engineering-strain percentage.

The finishing stage turns raw solutions into *reportable* ones:

* pairs whose coincidence lattice is finer than the reported supercell (that is,
  plain supercells of a smaller commensurate cell) are dropped,
* pairs related by ``(M, N) -> (G_t M K, G_b N K)`` for layer symmetries ``G_t``,
  ``G_b`` and a common unimodular ``K`` describe the very same bilayer, so only
  one representative of each such orbit is reported,
* the twist angle is folded into the fundamental range implied by the two layer
  symmetries, with the integer matrices moved along so that they still generate
  exactly the reported angle,
* the shared moire cell is Lagrange reduced, made right handed, and rotated so
  its first vector lies along ``+x``,
* both the relative and the per-layer realised principal strains are reported.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .gram_config import (
    SearchConfig,
    SearchResult,
    SymmetricBranchUnavailable,
    _REL,
)
from .gram_lattice import (
    _BottomIndex,
    _Table,
    _basis_table,
    _expand,
    _first_per_key,
    _fold_bases,
    _fold_sublattices,
    _gauge_group,
    _gram_of_basis,
    _gram_triples,
    _internal_length_scale,
    _lattice_vectors,
    _point_group,
    _proper_subgroup,
    _reduce_basis,
    _vector_orbit_representatives,
    _vector_table,
)
from .gram_lattice import _hermite_normal_form
from .gram_pairs import _canonical_pair_keys, _join_candidates, _shape_mask, _twist_angles
from .gram_report import _finalize, coincidence_index

__all__ = [
    "SearchConfig",
    "SearchResult",
    "SymmetricBranchUnavailable",
    "coincidence_index",
    "search",
    "symmetric_branch_applies",
]

def _general_search(config: SearchConfig) -> SearchResult:
    clock = time.perf_counter
    started = clock()
    lower, upper = config._band
    length_scale = _internal_length_scale(config)
    top_basis, top_gauge = _reduce_basis(config.top_basis / length_scale)
    bottom_basis, bottom_gauge = _reduce_basis(config.bottom_basis / length_scale)
    top_metric, bottom_metric = _gram_of_basis(top_basis), _gram_of_basis(bottom_basis)
    max_length = config.max_length / length_scale
    min_length = None if config.min_length is None else config.min_length / length_scale
    max_squared = max_length * max_length
    min_squared = 0.0 if min_length is None else min_length * min_length
    top_area = float(np.sqrt(np.linalg.det(top_metric)))
    bottom_area = float(np.sqrt(np.linalg.det(bottom_metric)))
    if config.fold_symmetry:
        top_group = _proper_subgroup(_gauge_group(config.top_group, top_gauge))
        bottom_group = _proper_subgroup(_gauge_group(config.bottom_group, bottom_gauge))
    else:
        top_group = bottom_group = np.eye(2, dtype=np.int64)[None, :, :]

    top_vectors = _vector_table(top_metric, top_basis, max_squared)
    bottom_vectors = _vector_table(bottom_metric, bottom_basis, upper * max_squared)
    after_vectors = clock()

    area_squared_max = None
    if config.max_atoms is not None:
        multiplicity_cap = config.max_atoms / (
            config.top_atoms + config.bottom_atoms * lower * top_area / bottom_area
        )
        area_squared_max = (multiplicity_cap * top_area) ** 2
    first_top = (
        np.nonzero(_vector_orbit_representatives(top_vectors.vectors, top_group))[0]
        if config.fold_symmetry
        else None
    )
    top = _basis_table(
        top_vectors,
        top_metric,
        partner=False,
        first_indices=first_top,
        area_squared_max=area_squared_max,
    )
    top_unfolded = len(top)
    top = top.take(_first_per_key(_hermite_normal_form(top.first, top.second)))
    if config.fold_symmetry and len(top_group) > 1:
        top = top.take(_fold_sublattices(top.first, top.second, top_group))
    top_after_fold = len(top)
    top = top.take(_shape_mask(top, config))
    if min_squared > 0.0:
        top = top.take(top.g11 >= min_squared * (1.0 - _REL))
    if config.max_atoms is not None:
        minimum_weight = (
            config.top_atoms + config.bottom_atoms * lower * top_area / bottom_area
        )
        top = top.take(top.index * minimum_weight <= config.max_atoms)
    after_top = clock()

    first_bottom = (
        np.nonzero(_vector_orbit_representatives(bottom_vectors.vectors, bottom_group))[0]
        if config.fold_symmetry
        else None
    )
    bottom = _basis_table(
        bottom_vectors,
        bottom_metric,
        partner=True,
        lower=lower,
        upper=upper,
        first_indices=first_bottom,
    )
    bottom_unfolded = len(bottom)
    if config.fold_symmetry and len(bottom_group) > 1:
        bottom = bottom.take(_fold_bases(bottom.first, bottom.second, bottom_group))
    bottom_after_fold = len(bottom)
    # Proven necessary transfers of the top shape/length/atom filters.
    keep = bottom.g22 <= (
        (upper / lower)
        * config.max_aspect_ratio**2
        * bottom.g11
        * (1.0 + _REL)
    )
    cosine_limit = max(
        abs(math.cos(math.radians(config.min_cell_angle_deg))),
        abs(math.cos(math.radians(config.max_cell_angle_deg))),
    )
    coefficient = (lower * cosine_limit + upper - lower) ** 2
    keep &= lower**2 * bottom.g12**2 <= (
        coefficient * bottom.g11 * bottom.g22 * (1.0 + _REL)
    )
    bottom = bottom.take(keep)
    if min_squared > 0.0:
        threshold = lower * min_squared * (1.0 - _REL)
        bottom = bottom.take((bottom.g11 >= threshold) & (bottom.g22 >= threshold))
    if config.max_atoms is not None:
        minimum_weight = (
            config.bottom_atoms + config.top_atoms * bottom_area / (upper * top_area)
        )
        bottom = bottom.take(bottom.index * minimum_weight <= config.max_atoms)
    after_bottom = clock()

    if len(top) == 0 or len(bottom) == 0:
        top_rows = bottom_rows = np.zeros(0, dtype=np.int64)
    else:
        index = _BottomIndex(bottom, lower, upper)
        top_rows, sorted_bottom_rows = _join_candidates(top, index, lower, upper)
        bottom_rows = index.order[sorted_bottom_rows]
    after_join = clock()
    if config.max_atoms is not None:
        exact_atoms = (
            top.index[top_rows] * config.top_atoms
            + bottom.index[bottom_rows] * config.bottom_atoms
            <= config.max_atoms
        )
        top_rows, bottom_rows = top_rows[exact_atoms], bottom_rows[exact_atoms]

    top_first, top_second = top.first[top_rows], top.second[top_rows]
    bottom_first, bottom_second = bottom.first[bottom_rows], bottom.second[bottom_rows]
    p11, p12, p22 = top.g11[top_rows], top.g12[top_rows], top.g22[top_rows]
    q11, q12, q22 = (
        bottom.g11[bottom_rows],
        bottom.g12[bottom_rows],
        bottom.g22[bottom_rows],
    )
    top_multiplicity, bottom_multiplicity = top.index[top_rows], bottom.index[bottom_rows]
    twist = _twist_angles(
        top_basis,
        bottom_basis,
        top_first,
        top_second,
        bottom_first,
        bottom_second,
    )
    top_matrices_reduced = np.stack([top_first, top_second], axis=2)
    bottom_matrices_reduced = np.stack([bottom_first, bottom_second], axis=2)
    pair_keys = _canonical_pair_keys(top_matrices_reduced, bottom_matrices_reduced)
    unique = _first_per_key(pair_keys)
    top_matrices_reduced = top_matrices_reduced[unique]
    bottom_matrices_reduced = bottom_matrices_reduced[unique]
    p11, p12, p22 = p11[unique], p12[unique], p22[unique]
    q11, q12, q22 = q11[unique], q12[unique], q22[unique]
    top_multiplicity = top_multiplicity[unique]
    bottom_multiplicity = bottom_multiplicity[unique]
    twist = twist[unique]
    top_matrices = top_gauge @ top_matrices_reduced
    bottom_matrices = bottom_gauge @ bottom_matrices_reduced
    stats = {
        "branch": "general",
        "n_top_rows": len(top),
        "n_bottom_rows": len(bottom),
        "n_top_rows_unfolded": top_unfolded,
        "n_bottom_rows_unfolded": bottom_unfolded,
        "n_top_rows_after_fold": top_after_fold,
        "n_bottom_rows_after_fold": bottom_after_fold,
        "group_order_top": int(len(top_group)),
        "group_order_bottom": int(len(bottom_group)),
        "n_shells_top": top_vectors.shell_count,
        "n_shells_bottom": bottom_vectors.shell_count,
        "t_vectors": after_vectors - started,
        "t_top_table": after_top - after_vectors,
        "t_bottom_table": after_bottom - after_top,
        "t_join": after_join - after_bottom,
        "t_finish": clock() - after_join,
        "t_total": clock() - started,
    }
    return _finalize(
        config,
        top_matrices,
        bottom_matrices,
        np.stack([p11, p12, p22], axis=1),
        np.stack([q11, q12, q22], axis=1),
        top_multiplicity,
        bottom_multiplicity,
        twist,
        stats,
        length_scale,
    )


def _right_handed_rotation(rotation: np.ndarray) -> np.ndarray:
    if rotation[1, 0] >= 0:
        return rotation.astype(np.int64)
    trace = rotation[0, 0] + rotation[1, 1]
    return np.array(
        [
            [trace - rotation[0, 0], -rotation[0, 1]],
            [-rotation[1, 0], trace - rotation[1, 1]],
        ],
        dtype=np.int64,
    )


def _rotation_generator(metric: np.ndarray, tolerance: float = 1e-10):
    group = _proper_subgroup(_point_group(metric, tolerance=tolerance))
    traces = group[:, 0, 0] + group[:, 1, 1]
    square = np.nonzero(traces == 0)[0]
    if square.size:
        return _right_handed_rotation(group[square[0]]), 0
    if np.any((traces == 1) | (traces == -1)):
        hexagonal = np.nonzero(traces == 1)[0]
        if hexagonal.size:
            return _right_handed_rotation(group[hexagonal[0]]), -1
    return None, None


def symmetric_branch_applies(config: SearchConfig) -> bool:
    """Return whether both layers have the same square or hexagonal rotation family."""
    length_scale = _internal_length_scale(config)
    top_basis, _ = _reduce_basis(config.top_basis / length_scale)
    bottom_basis, _ = _reduce_basis(config.bottom_basis / length_scale)
    top_rotation, top_kind = _rotation_generator(_gram_of_basis(top_basis))
    bottom_rotation, bottom_kind = _rotation_generator(_gram_of_basis(bottom_basis))
    return (
        top_rotation is not None
        and bottom_rotation is not None
        and top_kind == bottom_kind
    )


def _invariant_table(
    metric: np.ndarray,
    rotation: np.ndarray,
    radius_squared: float,
    fold: bool,
    group: np.ndarray | None = None,
):
    vectors, squared = _lattice_vectors(metric, radius_squared)
    if fold and len(vectors):
        folding_group = (
            _proper_subgroup(_point_group(metric, tolerance=1e-10))
            if group is None
            else _proper_subgroup(group)
        )
        keep = _vector_orbit_representatives(vectors, folding_group)
        vectors, squared = vectors[keep], squared[keep]
    rotated = vectors @ rotation.T
    index = np.abs(
        vectors[:, 0] * rotated[:, 1] - vectors[:, 1] * rotated[:, 0]
    )
    return vectors, rotated, squared, index


def _symmetric_search(config: SearchConfig) -> SearchResult:
    clock = time.perf_counter
    started = clock()
    lower, upper = config._band
    length_scale = _internal_length_scale(config)
    top_basis, top_gauge = _reduce_basis(config.top_basis / length_scale)
    bottom_basis, bottom_gauge = _reduce_basis(config.bottom_basis / length_scale)
    top_metric, bottom_metric = _gram_of_basis(top_basis), _gram_of_basis(bottom_basis)
    top_rotation, top_kind = _rotation_generator(top_metric)
    bottom_rotation, bottom_kind = _rotation_generator(bottom_metric)
    if top_rotation is None or bottom_rotation is None or top_kind != bottom_kind:
        raise SymmetricBranchUnavailable(
            "the symmetric branch requires both layers to be square or hexagonal "
            "with the same rotation order"
        )
    max_length = config.max_length / length_scale
    min_length = None if config.min_length is None else config.min_length / length_scale
    max_squared = max_length * max_length
    top_first, top_second, top_squared, top_index = _invariant_table(
        top_metric,
        top_rotation,
        max_squared,
        config.fold_symmetry,
        _gauge_group(config.top_group, top_gauge),
    )
    if min_length is not None:
        keep = top_squared >= min_length**2 * (1.0 - _REL)
        top_first, top_second = top_first[keep], top_second[keep]
        top_squared, top_index = top_squared[keep], top_index[keep]
    bottom_first, bottom_second, bottom_squared, bottom_index = _invariant_table(
        bottom_metric,
        bottom_rotation,
        upper * max_squared,
        config.fold_symmetry,
        _gauge_group(config.bottom_group, bottom_gauge),
    )
    after_tables = clock()
    low = np.searchsorted(
        bottom_squared, lower * top_squared * (1.0 - _REL), side="left"
    )
    high = np.searchsorted(
        bottom_squared, upper * top_squared * (1.0 + _REL), side="right"
    )
    top_rows, bottom_rows = _expand(low, high, np.arange(len(top_squared)))
    p, q = top_squared[top_rows], bottom_squared[bottom_rows]
    exact = (q >= lower * p) & (q <= upper * p)
    top_rows, bottom_rows = top_rows[exact], bottom_rows[exact]
    if config.max_atoms is not None:
        atom_keep = (
            top_index[top_rows] * config.top_atoms
            + bottom_index[bottom_rows] * config.bottom_atoms
            <= config.max_atoms
        )
        top_rows, bottom_rows = top_rows[atom_keep], bottom_rows[atom_keep]
    after_join = clock()
    top_first_result, top_second_result = top_first[top_rows], top_second[top_rows]
    bottom_first_result, bottom_second_result = (
        bottom_first[bottom_rows],
        bottom_second[bottom_rows],
    )
    p11, p12, p22 = _gram_triples(
        top_metric, top_first_result, top_second_result
    )
    q11, q12, q22 = _gram_triples(
        bottom_metric, bottom_first_result, bottom_second_result
    )
    shape_table = _Table(
        top_first_result,
        top_second_result,
        p11,
        p12,
        p22,
        top_index[top_rows],
    )
    shape_keep = _shape_mask(shape_table, config)
    top_first_result, top_second_result = (
        top_first_result[shape_keep],
        top_second_result[shape_keep],
    )
    bottom_first_result, bottom_second_result = (
        bottom_first_result[shape_keep],
        bottom_second_result[shape_keep],
    )
    p11, p12, p22 = p11[shape_keep], p12[shape_keep], p22[shape_keep]
    q11, q12, q22 = q11[shape_keep], q12[shape_keep], q22[shape_keep]
    top_multiplicity = top_index[top_rows][shape_keep]
    bottom_multiplicity = bottom_index[bottom_rows][shape_keep]
    top_cartesian = top_first_result @ top_basis.T
    bottom_cartesian = bottom_first_result @ bottom_basis.T
    twist = np.arctan2(
        top_cartesian[:, 0] * bottom_cartesian[:, 1]
        - top_cartesian[:, 1] * bottom_cartesian[:, 0],
        top_cartesian[:, 0] * bottom_cartesian[:, 0]
        + top_cartesian[:, 1] * bottom_cartesian[:, 1],
    )
    top_reduced = np.stack([top_first_result, top_second_result], axis=2)
    bottom_reduced = np.stack([bottom_first_result, bottom_second_result], axis=2)
    unique = _first_per_key(_canonical_pair_keys(top_reduced, bottom_reduced))
    top_reduced, bottom_reduced = top_reduced[unique], bottom_reduced[unique]
    p11, p12, p22 = p11[unique], p12[unique], p22[unique]
    q11, q12, q22 = q11[unique], q12[unique], q22[unique]
    top_multiplicity, bottom_multiplicity = (
        top_multiplicity[unique],
        bottom_multiplicity[unique],
    )
    twist = twist[unique]
    top_matrices = top_gauge @ top_reduced
    bottom_matrices = bottom_gauge @ bottom_reduced
    stats = {
        "branch": "symmetric",
        "symmetry_kind": "hexagonal" if top_kind == -1 else "square",
        "n_top_rows": int(len(top_squared)),
        "n_bottom_rows": int(len(bottom_squared)),
        "t_tables": after_tables - started,
        "t_join": after_join - after_tables,
        "t_finish": clock() - after_join,
        "t_total": clock() - started,
    }
    return _finalize(
        config,
        top_matrices,
        bottom_matrices,
        np.stack([p11, p12, p22], axis=1),
        np.stack([q11, q12, q22], axis=1),
        top_multiplicity,
        bottom_multiplicity,
        twist,
        stats,
        length_scale,
    )


def search(config: SearchConfig) -> SearchResult:
    """Search canonical bilayer candidates using the general or restricted engine."""
    if not isinstance(config, SearchConfig):
        raise TypeError("search expects a SearchConfig")
    return _symmetric_search(config) if config.symmetric else _general_search(config)
