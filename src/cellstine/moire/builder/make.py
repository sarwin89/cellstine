"""High-level bilayer superstructure generation stage."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import islice
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from ...core.provenance import stage_comment
from ...core.outputs import reserve_default_file
from . import generator as generator_backend

DEFAULT_OUTPUT_DIR = Path("output")


@dataclass
class MakeRun:
    output_path: Path
    selected_index: int
    angle_deg: float
    total_atoms: int
    contact_distance: float = float("nan")
    contact_species: tuple[str, str] | None = None
    structure_contact_distance: float = float("nan")
    notes: tuple[str, ...] = ()


def _limit_worker_threads() -> None:
    import os

    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, "1")


def _slug(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "structure"


def _generate_from_results_worker(
    task: tuple[str, str, str, dict | None, dict]
) -> MakeRun:
    results_file, top_poscar, bottom_poscar, record, options = task
    return _generate_from_record(results_file, top_poscar, bottom_poscar, record, **options)


def generate_from_results(
    results_file: str,
    *,
    index: int,
    interlayer_distance: float,
    output_path: str | None = None,
    output_dir: str | None = None,
    tolerance: int = 1,
    tolerance_float: float = 1e-4,
    zfix: float | None = None,
    top_c_repeat: int | None = None,
    bottom_c_repeat: int | None = None,
    vacuum: float | None = None,
) -> MakeRun:
    top_poscar, bottom_poscar, records, _payload = generator_backend.parse_results(results_file)
    by_index = {int(record["index"]): record for record in records}
    return _generate_from_record(
        results_file, top_poscar, bottom_poscar, by_index.get(index),
        index=index, interlayer_distance=interlayer_distance,
        output_path=output_path, output_dir=output_dir, tolerance=tolerance,
        tolerance_float=tolerance_float, zfix=zfix, top_c_repeat=top_c_repeat,
        bottom_c_repeat=bottom_c_repeat, vacuum=vacuum,
    )


def _generate_from_record(
    results_file: str,
    top_poscar: str,
    bottom_poscar: str,
    record: dict | None,
    *,
    index: int,
    interlayer_distance: float,
    output_path: str | None = None,
    output_dir: str | None = None,
    tolerance: int = 1,
    tolerance_float: float = 1e-4,
    zfix: float | None = None,
    top_c_repeat: int | None = None,
    bottom_c_repeat: int | None = None,
    vacuum: float | None = None,
) -> MakeRun:
    """Build one selected record without rereading the batch's result snapshot."""
    if record is None:
        raise ValueError(f"index {index} not found in {results_file}")

    resolved_top_c_repeat = int(top_c_repeat if top_c_repeat is not None else 1)
    resolved_bottom_c_repeat = int(bottom_c_repeat if bottom_c_repeat is not None else 1)

    lattice_out, positions_direct, counts, species, flags, contacts = generator_backend.build_supercell_with_report(
        top_poscar,
        bottom_poscar,
        record,
        interlayer_distance=float(interlayer_distance),
        tolerance=tolerance,
        tolerance_float=tolerance_float,
        zfix=zfix,
        repeat_top_c=resolved_top_c_repeat,
        repeat_bottom_c=resolved_bottom_c_repeat,
        vacuum=vacuum,
    )

    total_atoms = int(sum(counts))
    angle_deg = float(record["angle_deg"])
    if output_path is None:
        destination_dir = DEFAULT_OUTPUT_DIR.resolve() if output_dir is None else Path(output_dir).resolve()
        destination_dir.mkdir(parents=True, exist_ok=True)
        output_name = (
            f"stack_idx{index:03d}_ang{angle_deg:.4f}_atoms{total_atoms}_"
            f"{_slug(Path(bottom_poscar).stem)}-below_{_slug(Path(top_poscar).stem)}-above.vasp"
        )
        output_path = str(reserve_default_file(destination_dir / output_name))
    else:
        Path(output_path).resolve().parent.mkdir(parents=True, exist_ok=True)

    generator_backend.write_supercell_poscar(
        output_path,
        lattice_out,
        positions_direct,
        counts,
        species,
        flags,
        comment=stage_comment(
            "moire make",
            f"{Path(bottom_poscar).stem} below, {Path(top_poscar).stem} above",
            f"candidate {index}",
            f"twist {angle_deg:.4f} deg",
            f"{total_atoms} atoms",
        ),
    )

    contact = contacts.get("contact") or {}
    contact_species = contact.get("species")
    return MakeRun(
        output_path=Path(output_path).resolve(),
        selected_index=index,
        angle_deg=angle_deg,
        total_atoms=total_atoms,
        contact_distance=float(contacts.get("contact_distance", float("nan"))),
        contact_species=None if not contact_species else (str(contact_species[0]), str(contact_species[1])),
        structure_contact_distance=float(
            contacts.get("structure_contact_distance", float("nan"))
        ),
        notes=tuple(contacts.get("notes", ())),
    )


def _ordered_parallel_results(executor, worker, tasks, *, max_pending: int):
    """Yield requested order while bounding queued payloads and futures.

    An error or an early close cancels work which has not started. The owning
    executor context waits for active writers; completed outputs are not retried.
    """
    task_iterator = iter(tasks)
    pending = deque()
    try:
        for task in islice(task_iterator, max_pending):
            pending.append(executor.submit(worker, task))
        while pending:
            yield pending.popleft().result()
            for task in islice(task_iterator, 1):
                pending.append(executor.submit(worker, task))
    finally:
        for future in pending:
            future.cancel()


def generate_many_from_results(
    results_file: str,
    *,
    indexes: Sequence[int],
    interlayer_distance: float,
    output_path: str | None = None,
    output_dir: str | None = None,
    tolerance: int = 1,
    tolerance_float: float = 1e-4,
    zfix: float | None = None,
    top_c_repeat: int | None = None,
    bottom_c_repeat: int | None = None,
    vacuum: float | None = None,
    workers: int = 1,
    parallel_backend: str = "process",
) -> List[MakeRun]:
    """Calculate and write independent structures with explicit bounded concurrency."""
    if parallel_backend not in ("process", "thread"):
        raise ValueError("parallel_backend must be process or thread")
    resolved_indexes = [int(index) for index in indexes]
    resolved_workers = max(1, int(workers))
    if len(resolved_indexes) > 1 and output_path is not None:
        raise ValueError("use output_path only when generating a single index")

    if not resolved_indexes:
        return []
    top_poscar, bottom_poscar, records, _payload = generator_backend.parse_results(results_file)
    by_index = {int(record["index"]): record for record in records}
    options = dict(
        interlayer_distance=interlayer_distance, output_path=output_path,
        output_dir=output_dir, tolerance=tolerance, tolerance_float=tolerance_float,
        zfix=zfix, top_c_repeat=top_c_repeat, bottom_c_repeat=bottom_c_repeat, vacuum=vacuum,
    )
    # Each process receives only its selected candidate, never the entire result
    # document. Missing indexes remain task errors, preserving ordered failures.
    task_inputs = (
        (results_file, top_poscar, bottom_poscar, by_index.get(index), {**options, "index": index})
        for index in resolved_indexes
    )
    if resolved_workers <= 1 or len(resolved_indexes) <= 1:
        return [_generate_from_results_worker(task) for task in task_inputs]
    effective_workers = min(resolved_workers, len(resolved_indexes))
    try:
        if parallel_backend == "thread":
            # Threads share the caller's numerical runtime and must not mutate
            # its environment or native-library thread configuration.
            executor = ThreadPoolExecutor(max_workers=effective_workers)
        else:
            executor = ProcessPoolExecutor(max_workers=effective_workers, initializer=_limit_worker_threads)
    except OSError:
        # Construction has dispatched no tasks, so a synchronous retry is safe.
        return [_generate_from_results_worker(task) for task in task_inputs]
    with executor:
        # Once dispatch begins, an OSError may be a candidate's write failure;
        # retrying the whole batch would duplicate already completed outputs.
        return list(_ordered_parallel_results(
            executor, _generate_from_results_worker, task_inputs, max_pending=2 * effective_workers
        ))


def make(**kwargs):
    return generate_many_from_results(**kwargs)
