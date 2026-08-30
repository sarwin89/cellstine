"""High-level bilayer superstructure generation stage."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from ...core.provenance import stage_comment
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
    task: tuple[str, int, float, str | None, int, float, float | None, int | None, int | None, float | None]
) -> MakeRun:
    (
        task_results,
        task_index,
        task_interlayer,
        task_output_dir,
        task_tolerance,
        task_tolerance_float,
        task_zfix,
        task_top_repeat,
        task_bottom_repeat,
        task_vacuum,
    ) = task
    return generate_from_results(
        task_results,
        index=task_index,
        interlayer_distance=task_interlayer,
        output_dir=task_output_dir,
        tolerance=task_tolerance,
        tolerance_float=task_tolerance_float,
        zfix=task_zfix,
        top_c_repeat=task_top_repeat,
        bottom_c_repeat=task_bottom_repeat,
        vacuum=task_vacuum,
    )


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
    if index not in by_index:
        raise ValueError(f"index {index} not found in {results_file}")

    resolved_top_c_repeat = int(top_c_repeat if top_c_repeat is not None else 1)
    resolved_bottom_c_repeat = int(bottom_c_repeat if bottom_c_repeat is not None else 1)

    record = by_index[index]
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
        output_path = str(destination_dir / output_name)
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
) -> List[MakeRun]:
    resolved_indexes = [int(index) for index in indexes]
    resolved_workers = max(1, int(workers))
    if len(resolved_indexes) > 1 and output_path is not None:
        raise ValueError("use output_path only when generating a single index")

    if resolved_workers <= 1 or len(resolved_indexes) <= 1:
        runs: List[MakeRun] = []
        for index in resolved_indexes:
            run = generate_from_results(
                results_file,
                index=index,
                interlayer_distance=interlayer_distance,
                output_path=output_path if len(resolved_indexes) == 1 else None,
                output_dir=output_dir,
                tolerance=tolerance,
                tolerance_float=tolerance_float,
                zfix=zfix,
                top_c_repeat=top_c_repeat,
                bottom_c_repeat=bottom_c_repeat,
                vacuum=vacuum,
            )
            runs.append(run)
        return runs

    task_inputs = [
        (
            results_file,
            int(index),
            float(interlayer_distance),
            output_dir,
            tolerance,
            tolerance_float,
            zfix,
            top_c_repeat,
            bottom_c_repeat,
            vacuum,
        )
        for index in resolved_indexes
    ]

    try:
        with ProcessPoolExecutor(max_workers=resolved_workers, initializer=_limit_worker_threads) as executor:
            return list(executor.map(_generate_from_results_worker, task_inputs))
    except (OSError, PermissionError):
        return [_generate_from_results_worker(task) for task in task_inputs]


def make(**kwargs):
    return generate_many_from_results(**kwargs)
