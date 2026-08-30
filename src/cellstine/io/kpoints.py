"""Native VASP KPOINTS reader and writer.

Two of the KPOINTS layouts matter for plane-wave structure generation and both
are supported here:

*automatic* -- a zero on the second line, then ``Gamma`` or ``Monkhorst``, the
three divisions, and an optional shift in grid steps.  The code generates the
mesh itself, so nothing is lost by writing it this way and it stays readable.

*explicit* -- the number of points on the second line, then ``Reciprocal`` (or
``Cartesian``) and one line per point with its weight.  This is the form a
symmetry-reduced list needs, and it is what :class:`cellstine.core.reciprocal.
KpointMesh` writes when its irreducible points are requested.

*line mode* -- the number of points *per segment* on the second line, the word
``Line-mode``, the coordinate word, and then the two ends of every segment, one
blank line between segments.  This is what a band structure is asked for, and
:func:`write_band_path` writes the path built by
:mod:`cellstine.core.kpath` in it, each end carrying the name of the
high-symmetry point as a trailing comment.

Weights are written as integers whenever they are integers, which is the case
for an orbit count, so a reduced mesh round-trips through the file exactly.  The
reader accepts either spelling of the mode word and any capitalisation, which is
all VASP itself reads.

One subtlety decides whether an automatic file means the mesh it was asked to
mean.  The mesh point of ``KPOINTS`` is ``(i + s) / n`` along each axis, where
``s`` is the shift written on the last line, *in grid steps* -- but the word
``Monkhorst`` already carries half a step of its own along every axis with an
even division.  A mesh that is half-shifted therefore has two faithful spellings
and one wrong one: ``Monkhorst`` with no shift line, or ``Gamma`` with a shift of
one half, but never ``Monkhorst`` with a shift of one half, which is a whole
step and lands back on the Gamma-centred grid.  :func:`write_mesh` picks a
spelling that reproduces the mesh, and :meth:`KpointsFile.total_shift` folds the
mode word back into the offset so a file can be compared against the mesh it
came from.  The statements behind this are in ``RequestProject/MeshShift.lean``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..core.reciprocal import KpointMesh, mesh_points, mesh_shift

__all__ = [
    "KpointsFile",
    "write_automatic_kpoints",
    "write_explicit_kpoints",
    "write_line_mode_kpoints",
    "write_band_path",
    "write_mesh",
    "read_kpoints",
]


@dataclass(frozen=True)
class KpointsFile:
    """A parsed KPOINTS file."""

    comment: str
    mode: str
    divisions: tuple[int, int, int] | None = None
    shift: tuple[float, float, float] | None = None
    points: np.ndarray | None = None
    weights: np.ndarray | None = None
    coordinate_mode: str | None = None
    line_divisions: int | None = None
    endpoints: np.ndarray | None = None
    labels: tuple[str, ...] | None = None

    @property
    def point_count(self) -> int:
        """Return the number of explicit points, or zero for an automatic mesh."""

        return 0 if self.points is None else int(len(self.points))

    @property
    def total_shift(self) -> tuple[float, float, float]:
        """Return the offset of an automatic mesh in grid steps, mode word included.

        ``Monkhorst`` is half a step along every even axis before the shift line
        is added, so this is the offset VASP will actually apply.  A whole step
        is the mesh itself, so the result is folded back into ``[0, 1)``.
        """

        if self.divisions is None:
            raise ValueError("only an automatic mesh has a grid-step shift")
        written = np.asarray(self.shift or (0.0, 0.0, 0.0), dtype=float)
        implied = np.asarray(
            mesh_shift(self.divisions, "monkhorst" if self.mode == "monkhorst" else "gamma"),
            dtype=float,
        )
        return tuple(float(value % 1.0) for value in written + implied)

    def line_points(self) -> np.ndarray:
        """Return the points of a line-mode file, in the order VASP samples them.

        Every segment carries :attr:`line_divisions` points including both ends,
        and, as in the file, the shared end of two segments is written twice --
        which is exactly the list of eigenvalues a band structure comes back
        with.
        """

        if self.endpoints is None or self.line_divisions is None:
            raise ValueError("only a line-mode file has segment points")
        count = int(self.line_divisions)
        ends = np.asarray(self.endpoints, dtype=float).reshape(-1, 2, 3)
        fractions = np.linspace(0.0, 1.0, count)[:, None]
        pieces = [start + fractions * (end - start) for start, end in ends]
        return np.concatenate(pieces, axis=0)

    def mesh_points(self) -> np.ndarray:
        """Return the points this file stands for, in fractional reciprocal coordinates.

        For an explicit list that is the list itself; for an automatic mesh it is
        the grid VASP would build, so the two layouts can be compared directly.
        """

        if self.mode == "line":
            return self.line_points()
        if self.points is not None:
            return np.asarray(self.points, dtype=float)
        return mesh_points(self.divisions, self.total_shift)


def _format_number(value: float) -> str:
    return f"{float(value): .10f}"


def _format_weight(value: float) -> str:
    number = float(value)
    if abs(number - round(number)) <= 1e-12:
        return f"{int(round(number)):d}"
    return f"{number:.12g}"


def write_automatic_kpoints(
    path: str | Path,
    divisions: Sequence[int],
    *,
    shift: Sequence[float] = (0.0, 0.0, 0.0),
    gamma_centred: bool = True,
    comment: str = "Automatic mesh",
) -> Path:
    """Write an automatic mesh KPOINTS file and return its path."""

    counts = [int(item) for item in np.asarray(divisions).ravel()]
    if len(counts) != 3 or any(item < 1 for item in counts):
        raise ValueError("an automatic mesh needs three divisions of at least one")
    offsets = [float(item) for item in np.asarray(shift, dtype=float).ravel()]
    if len(offsets) != 3:
        raise ValueError("a mesh shift needs three components")
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        str(comment).strip() or "Automatic mesh",
        "0",
        "Gamma" if gamma_centred else "Monkhorst-Pack",
        " ".join(str(item) for item in counts),
        " ".join(f"{item:.10f}" for item in offsets),
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def write_explicit_kpoints(
    path: str | Path,
    points: Sequence[Sequence[float]],
    weights: Sequence[float],
    *,
    comment: str = "Explicit k-points",
    coordinate_mode: str = "reciprocal",
) -> Path:
    """Write an explicit list of points with weights and return its path."""

    array = np.asarray(points, dtype=float).reshape(-1, 3)
    values = np.asarray(weights, dtype=float).ravel()
    if len(array) == 0:
        raise ValueError("an explicit KPOINTS file needs at least one point")
    if len(values) != len(array):
        raise ValueError("every k-point needs exactly one weight")
    if np.any(values <= 0.0):
        raise ValueError("every k-point weight must be positive")
    name = str(coordinate_mode).lower()
    if name.startswith("r"):
        header = "Reciprocal"
    elif name.startswith("c") or name.startswith("k"):
        header = "Cartesian"
    else:
        raise ValueError("the coordinate mode must be 'reciprocal' or 'cartesian'")
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(comment).strip() or "Explicit k-points", str(len(array)), header]
    for point, weight in zip(array, values):
        coordinates = " ".join(_format_number(value) for value in point)
        lines.append(f"{coordinates}  {_format_weight(weight)}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _coordinate_header(coordinate_mode: str) -> str:
    name = str(coordinate_mode).lower()
    if name.startswith("r"):
        return "Reciprocal"
    if name.startswith("c") or name.startswith("k"):
        return "Cartesian"
    raise ValueError("the coordinate mode must be 'reciprocal' or 'cartesian'")


def write_line_mode_kpoints(
    path: str | Path,
    segments: Sequence[tuple[tuple[str, Sequence[float]], tuple[str, Sequence[float]]]],
    divisions: int,
    *,
    comment: str = "Band path",
    coordinate_mode: str = "reciprocal",
) -> Path:
    """Write a line-mode KPOINTS file and return its path.

    ``segments`` is a sequence of ``((start_label, start_point), (end_label,
    end_point))`` pairs and ``divisions`` the number of points sampled on each
    of them, both ends included.  A segment of zero length is refused: it would
    ask the code for the same eigenvalues twice and leaves a band plot with a
    step of zero width.
    """

    count = int(divisions)
    if count < 2:
        raise ValueError("a line-mode segment needs at least two points")
    if not segments:
        raise ValueError("a line-mode file needs at least one segment")
    header = _coordinate_header(coordinate_mode)
    lines = [str(comment).strip() or "Band path", str(count), "Line-mode", header]
    for index, ((start_label, start), (end_label, end)) in enumerate(segments):
        first = np.asarray(start, dtype=float).reshape(3)
        second = np.asarray(end, dtype=float).reshape(3)
        if float(np.linalg.norm(second - first)) <= 1e-12:
            raise ValueError(
                f"segment {index + 1} runs from {start_label} to {end_label}, which is the same point"
            )
        if index:
            lines.append("")
        lines.append(" ".join(_format_number(value) for value in first) + f"  ! {start_label}")
        lines.append(" ".join(_format_number(value) for value in second) + f"  ! {end_label}")
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def write_band_path(
    path: str | Path,
    band_path,
    *,
    divisions: int | None = None,
    spacing: float | None = None,
    comment: str | None = None,
) -> Path:
    """Write a :class:`cellstine.core.kpath.BandPath` in line mode.

    Either ``divisions`` or ``spacing`` fixes how finely the path is sampled;
    ``spacing`` is a largest step in inverse angstrom and is turned into the
    division count that meets it on every segment.
    """

    if divisions is None and spacing is None:
        raise ValueError("a band path needs either a division count or a spacing")
    if divisions is not None and spacing is not None:
        raise ValueError("give a division count or a spacing, not both")
    count = int(divisions) if divisions is not None else band_path.divisions_for_spacing(float(spacing))
    segments: list[tuple[tuple[str, np.ndarray], tuple[str, np.ndarray]]] = []
    for run, run_points in zip(band_path.walk, band_path.walk_points):
        coordinates = np.asarray(run_points, dtype=float)
        for position in range(len(run) - 1):
            segments.append(
                ((run[position], coordinates[position]), (run[position + 1], coordinates[position + 1]))
            )
    label = comment or f"{band_path.bravais} band path {band_path.path_string()}"
    return write_line_mode_kpoints(path, segments, count, comment=label)


def write_mesh(
    path: str | Path,
    mesh: KpointMesh,
    *,
    explicit: bool | None = None,
    comment: str | None = None,
) -> Path:
    """Write a :class:`KpointMesh`, reduced or whole, and return its path.

    ``explicit`` chooses the layout: the irreducible list with weights, or the
    automatic mesh line that lets the code rebuild the same grid.  The default
    writes the list exactly when the reduction actually removed points, since an
    automatic line would then silently discard the saving.
    """

    reduced = mesh.point_count < mesh.full_point_count
    use_list = reduced if explicit is None else bool(explicit)
    label = comment or (
        f"{mesh.divisions[0]}x{mesh.divisions[1]}x{mesh.divisions[2]} mesh, "
        f"{mesh.point_count} of {mesh.full_point_count} points"
    )
    if use_list:
        return write_explicit_kpoints(
            path, mesh.points, mesh.weights.astype(float), comment=label
        )
    offset = np.asarray(mesh.shift, dtype=float)
    monkhorst = np.asarray(mesh_shift(mesh.divisions, "monkhorst"), dtype=float)
    if np.allclose(offset % 1.0, monkhorst) and np.any(monkhorst):
        # The mode word already carries this offset; writing it again would add a
        # whole grid step and hand back the Gamma-centred mesh.
        gamma_centred, written_shift = False, (0.0, 0.0, 0.0)
    else:
        gamma_centred, written_shift = True, tuple(float(value) for value in offset)
    return write_automatic_kpoints(
        path,
        mesh.divisions,
        shift=written_shift,
        gamma_centred=gamma_centred,
        comment=label,
    )


def _tokens(line: str) -> list[str]:
    return line.replace(",", " ").split()


def read_kpoints(path: str | Path) -> KpointsFile:
    """Read a KPOINTS file written in either supported layout."""

    text = Path(path).read_text(encoding="utf-8").splitlines()
    if len(text) < 3:
        raise ValueError("a KPOINTS file needs at least a comment, a count, and a mode")
    comment = text[0].strip()
    tokens = _tokens(text[1])
    if not tokens:
        raise ValueError("the second line of a KPOINTS file must hold the number of points")
    count = int(float(tokens[0]))
    mode_word = (text[2].strip() or "?")[0].lower()
    if count == 0:
        if mode_word == "g":
            mode = "gamma"
        elif mode_word == "m":
            mode = "monkhorst"
        else:
            raise ValueError("an automatic KPOINTS mesh must say 'Gamma' or 'Monkhorst'")
        division_tokens = _tokens(text[3]) if len(text) > 3 else []
        if len(division_tokens) < 3:
            raise ValueError("an automatic KPOINTS mesh needs three divisions")
        divisions = tuple(int(float(item)) for item in division_tokens[:3])
        shift = (0.0, 0.0, 0.0)
        if len(text) > 4:
            shift_tokens = _tokens(text[4])
            if len(shift_tokens) >= 3:
                shift = tuple(float(item) for item in shift_tokens[:3])
        return KpointsFile(comment=comment, mode=mode, divisions=divisions, shift=shift)
    if mode_word == "l":
        coordinate_word = (text[3].strip() or "?")[0].lower() if len(text) > 3 else "?"
        if coordinate_word not in {"r", "c", "k"}:
            raise ValueError("a line-mode KPOINTS file must say 'Reciprocal' or 'Cartesian'")
        coordinate_mode = "reciprocal" if coordinate_word == "r" else "cartesian"
        ends: list[list[float]] = []
        names: list[str] = []
        for line in text[4:]:
            body, _, remark = line.partition("!")
            entries = _tokens(body)
            if len(entries) < 3:
                continue
            ends.append([float(value) for value in entries[:3]])
            names.append(remark.strip())
        if len(ends) < 2 or len(ends) % 2:
            raise ValueError("a line-mode KPOINTS file needs both ends of every segment")
        return KpointsFile(
            comment=comment,
            mode="line",
            coordinate_mode=coordinate_mode,
            line_divisions=count,
            endpoints=np.asarray(ends, dtype=float),
            labels=tuple(names),
        )
    if mode_word in {"r", "c", "k"}:
        coordinate_mode = "reciprocal" if mode_word == "r" else "cartesian"
    else:
        raise ValueError("an explicit KPOINTS list must say 'Reciprocal' or 'Cartesian'")
    points: list[list[float]] = []
    weights: list[float] = []
    for line in text[3:]:
        entries = _tokens(line)
        if len(entries) < 4:
            continue
        points.append([float(value) for value in entries[:3]])
        weights.append(float(entries[3]))
        if len(points) == count:
            break
    if len(points) != count:
        raise ValueError("the KPOINTS file holds fewer points than it promises")
    return KpointsFile(
        comment=comment,
        mode="explicit",
        points=np.asarray(points, dtype=float),
        weights=np.asarray(weights, dtype=float),
        coordinate_mode=coordinate_mode,
    )
