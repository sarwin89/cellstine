import shutil
import sys
import unittest
from importlib.util import find_spec
from io import StringIO
from pathlib import Path
from unittest import mock
from uuid import uuid4

import numpy as np

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src")))

import cellstine
import moire_cli
from cellstine.cli import main as cli_main
from cellstine.cli.interactive import runner as interactive_cli
from cellstine.core.previews import format_adsorption_sites
from cellstine.defect.workflow import Defect as DefectWorkflow
from cellstine.io import native as io
from cellstine.adsorbate import molecule
from cellstine.adsorbate.molecule import Molecule as MoleculeWorkflow
from cellstine.interface.surface import backend as surface
from cellstine.interface.surface.surface import Surface as InterfaceSurface, _stacking_sequence
from cellstine.interface.workflow.interface import Interface as InterfaceWorkflow, parse_miller_notation
from cellstine.moire.builder import make
from cellstine.moire.search import find, lattice
from cellstine.visualize.backends.matplotlib import _marker_size
from cellstine.moire.moire import Moire as MoireWorkflow
from cellstine.symmetry.symmetry import Symmetry as SymmetryWorkflow


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / 'input'


def _sample_path(filename: str) -> str:
    input_path = INPUT_DIR / filename
    if input_path.exists():
        return str(input_path)
    fallback_path = BASE_DIR / filename
    return str(fallback_path)


MOS2_PATH = _sample_path('mos2.vasp')


def _simple_adsorbate_stack():
    lattice_out = np.diag([8.0, 8.0, 20.0])
    positions_cartesian = np.array(
        [
            [1.0, 1.0, 1.0],
            [4.0, 4.0, 1.5],
            [2.5, 3.0, 4.85],
            [3.5, 3.0, 6.0],
        ],
        dtype=float,
    )
    return (
        lattice_out,
        io.cartesian_to_direct(positions_cartesian, lattice_out),
        [2, 1, 1],
        ["Cu", "C", "O"],
        None,
    )


class MoireToolkitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mos2 = io.read_poscar(MOS2_PATH)

    def test_native_gram_find_to_json_make(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            run = find.run_find(
                top_poscar=MOS2_PATH,
                bottom_poscar=MOS2_PATH,
                max_length=4.0,
                top_strain=0.01,
                bottom_strain=0.01,
                max_atoms=200,
                fold_symmetry=False,
                output_root=str(temp_root / "search"),
            )
            self.assertEqual(run.result_path.suffix, ".json")
            self.assertIn('"schema": "cellstine.moire.gram"', run.result_path.read_text(encoding="utf-8"))

            built = make.generate_from_results(
                str(run.result_path),
                index=1,
                interlayer_distance=3.35,
                output_path=str(temp_root / "stack.vasp"),
            )
            structure = io.read_poscar(str(built.output_path))
            self.assertEqual(structure.natoms, int(run.candidates[0]["atom_count"]))
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_moire_translate_shifts_only_the_top_group(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            stacked_path = temp_root / "stacked_simple.vasp"
            io.write_poscar(
                str(stacked_path),
                np.diag([4.0, 4.0, 12.0]),
                np.array(
                    [
                        [0.1, 0.1, 0.10],
                        [0.6, 0.6, 0.18],
                        [0.2, 0.2, 0.78],
                        [0.7, 0.7, 0.86],
                    ],
                    dtype=float,
                ),
                [2, 2],
                ["A", "B"],
                comment="simple stacked structure",
                positions_are_cartesian=False,
            )

            moire_tool = MoireWorkflow(
                runs_root=str(temp_root / "runs"),
                output_root=str(temp_root / "output"),
            )
            shifted = moire_tool.translate(
                poscar_path=str(stacked_path),
                shift_direct=[0.25, 0.0, 0.0],
            )
            shifted_record = io.read_poscar(str(shifted.artifacts["output_poscar"]))
            self.assertTrue(
                np.allclose(
                    shifted_record.positions_direct[:2],
                    np.array([[0.1, 0.1, 0.10], [0.6, 0.6, 0.18]]),
                )
            )
            self.assertTrue(
                np.allclose(
                    shifted_record.positions_direct[2:, 0],
                    np.array([0.45, 0.95]),
                )
            )
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_cli_help_text_mentions_native_gram_controls(self):
        parser = moire_cli.build_parser()
        moire_group = parser._subparsers._group_actions[0].choices["moire"]
        help_text = moire_group._subparsers._group_actions[0].choices["find"].format_help()
        self.assertIn("--max-length", help_text)
        self.assertIn("--top-strain", help_text)
        self.assertIn("--bottom-strain", help_text)
        self.assertIn("principal logarithmic strain", help_text)
        self.assertNotIn("--nindex", help_text)
        self.assertNotIn("--angles", help_text)

    def test_cli_help_text_mentions_grouped_workflows_and_subcommands(self):
        parser = moire_cli.build_parser()
        choices = parser._subparsers._group_actions[0].choices
        self.assertIn("moire", choices)
        self.assertIn("adsorbate", choices)
        self.assertIn("interface", choices)
        self.assertIn("symmetry", choices)
        self.assertIn("defect", choices)

        moire_group = choices["moire"]
        moire_choices = moire_group._subparsers._group_actions[0].choices
        self.assertIn("find", moire_choices)
        self.assertIn("make", moire_choices)
        self.assertIn("visualize", moire_choices)
        self.assertNotIn("findn", moire_choices)
        self.assertNotIn("maken", moire_choices)

        adsorbate_choices = choices["adsorbate"]._subparsers._group_actions[0].choices
        self.assertIn("place", adsorbate_choices)
        self.assertIn("assemble", adsorbate_choices)

        interface_choices = choices["interface"]._subparsers._group_actions[0].choices
        self.assertIn("surface", interface_choices)
        self.assertIn("sites", interface_choices)
        self.assertIn("build", interface_choices)
        self.assertIn("match", interface_choices)

        defect_choices = choices["defect"]._subparsers._group_actions[0].choices
        self.assertIn("analyse", defect_choices)
        self.assertIn("generate", defect_choices)
        self.assertIn("preview", defect_choices)

        symmetry_choices = choices["symmetry"]._subparsers._group_actions[0].choices
        self.assertIn("analyse", symmetry_choices)
        self.assertIn("reduce", symmetry_choices)
        self.assertIn("lattice-reduce", symmetry_choices)

    def test_cellstine_package_exports_public_classes(self):
        self.assertEqual(cellstine.__version__, "4.0.0")
        self.assertTrue(hasattr(cellstine, "Moire"))
        self.assertTrue(hasattr(cellstine, "Molecule"))
        self.assertTrue(hasattr(cellstine, "Interface"))
        self.assertTrue(hasattr(cellstine, "Defect"))
        self.assertTrue(hasattr(cellstine, "Symmetry"))

    def test_documented_public_api_imports_load_from_canonical_modules(self):
        from cellstine import Defect, Interface, Moire, Surface, Visualize

        self.assertEqual(Moire.__module__, "cellstine.moire.moire")
        self.assertEqual(Surface.__module__, "cellstine.interface.surface.surface")
        self.assertEqual(Interface.__module__, "cellstine.interface.workflow.interface")
        self.assertEqual(Defect.__module__, "cellstine.defect.workflow")
        self.assertEqual(Visualize.__module__, "cellstine.visualize.visualize")

    def test_docs_reference_current_entrypoints_and_imports(self):
        readme_text = (BASE_DIR / "README.md").read_text(encoding="utf-8")
        guide_text = (BASE_DIR / "USAGE_GUIDE.md").read_text(encoding="utf-8")

        self.assertNotIn("top-level `moire` Python package has been retired", guide_text)
        self.assertNotIn("compatibility layer retired", readme_text)
        self.assertIn("cellstine symmetry", readme_text)
        self.assertIn("from cellstine import Symmetry", guide_text)

    def test_packaging_metadata_targets_src_layout_and_console_entrypoint(self):
        pyproject_text = (BASE_DIR / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn('description = "CELLSTINE: moire, adsorbate, interface, symmetry, and defect workflows for VASP structures."', pyproject_text)
        self.assertIn('cellstine = "cellstine.cli.main:main"', pyproject_text)
        self.assertIn('where = ["src"]', pyproject_text)
        self.assertIn('include = ["cellstine*"]', pyproject_text)

    def test_cellstine_source_tree_has_domain_hierarchy(self):
        package_root = BASE_DIR / "src" / "cellstine"
        expected_directories = [
            package_root / "moire" / "search",
            package_root / "moire" / "builder",
            package_root / "moire" / "transform",
            package_root / "adsorbate" / "placement",
            package_root / "adsorbate" / "transform",
            package_root / "interface" / "surface",
            package_root / "interface" / "workflow",
            package_root / "visualize" / "backends",
            package_root / "visualize" / "results",
            package_root / "cli" / "interactive",
        ]

        missing = [str(path.relative_to(package_root)) for path in expected_directories if not path.is_dir()]

        self.assertEqual(missing, [])

    def test_old_flat_domain_modules_are_not_left_behind(self):
        package_root = BASE_DIR / "src" / "cellstine"
        old_flat_modules = [
            package_root / "interface" / "surface_backend.py",
            package_root / "interface" / "surface.py",
            package_root / "interface" / "interface.py",
            package_root / "moire" / "finder.py",
            package_root / "moire" / "generator.py",
            package_root / "moire" / "lattice.py",
            package_root / "adsorbate" / "operations.py",
            package_root / "visualize" / "results_plotly.py",
            package_root / "visualize" / "matplotlib_backend.py",
            package_root / "visualize" / "plotly_backend.py",
            package_root / "cli" / "interactive.py",
        ]

        leftovers = [str(path.relative_to(package_root)) for path in old_flat_modules if path.exists()]

        self.assertEqual(leftovers, [])

    def test_subpackage_inits_are_not_reexport_barrels(self):
        package_root = BASE_DIR / "src" / "cellstine"
        barrel_inits = []
        for init_path in package_root.rglob("__init__.py"):
            if init_path == package_root / "__init__.py":
                continue
            text = init_path.read_text(encoding="utf-8")
            executable_lines = [
                line.strip()
                for line in text.splitlines()
                if line.strip() and not line.strip().startswith("#") and not line.strip().startswith('"""')
            ]
            import_lines = [line for line in executable_lines if line.startswith(("from .", "from .."))]
            if import_lines:
                barrel_inits.append(str(init_path.relative_to(package_root)))

        self.assertEqual(barrel_inits, [])

    def test_removed_top_level_moire_package_is_not_present(self):
        self.assertFalse((BASE_DIR / "moire").exists())

    def test_group_help_mentions_guided_mode(self):
        parser = moire_cli.build_parser()
        moire_help = parser._subparsers._group_actions[0].choices["moire"].format_help()
        self.assertIn("guided workflow", moire_help)

    def test_adsorbate_place_help_mentions_substrate_fit_options(self):
        parser = moire_cli.build_parser()
        adsorbate_group = parser._subparsers._group_actions[0].choices["adsorbate"]
        place_help = adsorbate_group._subparsers._group_actions[0].choices["place"].format_help()
        self.assertIn("substrate-supercell-matrix", place_help)
        self.assertIn("auto-repeat-substrate", place_help)
        self.assertIn("fit-padding", place_help)

    def test_dispatch_namespace_enters_group_guided_mode_when_stage_is_missing(self):
        namespace = type("Namespace", (), {"version": False, "group": "moire", "stage": None})()
        with mock.patch("cellstine.cli.interactive.runner.run_interactive", return_value=0) as patched:
            result = cli_main.dispatch_namespace(namespace)
        self.assertEqual(result, 0)
        patched.assert_called_once_with(group="moire")

    def test_compact_miller_notation_is_supported(self):
        self.assertEqual(parse_miller_notation("111"), (1, 1, 1))
        self.assertEqual(parse_miller_notation("001"), (0, 0, 1))
        self.assertEqual(parse_miller_notation("111x"), (1, 1, -1))
        self.assertEqual(parse_miller_notation("1x11"), (-1, 1, 1))
        self.assertEqual(parse_miller_notation("1,1,2x"), (1, 1, -2))

    def test_interactive_file_suggestions_prioritise_prompt_roots(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        input_dir = temp_root / "input"
        output_dir = temp_root / "output"
        input_dir.mkdir(parents=True, exist_ok=False)
        output_dir.mkdir(parents=True, exist_ok=False)
        try:
            input_path = input_dir / "source.vasp"
            output_path = output_dir / "generated.vasp"
            input_path.write_text("input", encoding="utf-8")
            output_path.write_text("output", encoding="utf-8")

            candidates = interactive_cli._find_candidates(("*.vasp",), (input_dir, output_dir))
            self.assertEqual(candidates[0], input_path.resolve())

            candidates = interactive_cli._find_candidates(("*.vasp",), (output_dir, input_dir))
            self.assertEqual(candidates[0], output_path.resolve())
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_interactive_choice_accepts_back_shortcut(self):
        with mock.patch("builtins.input", return_value="b"), mock.patch("sys.stdout", new_callable=StringIO):
            with self.assertRaises(interactive_cli._BackInteractive):
                interactive_cli._choice(
                    "Choose a workflow",
                    [{"key": "moire", "label": "Moire"}],
                )

    def test_interactive_choice_can_disable_back_shortcut(self):
        with mock.patch("builtins.input", side_effect=["b", "1"]), mock.patch("sys.stdout", new_callable=StringIO):
            choice = interactive_cli._choice(
                "Top-level workflow",
                [{"key": "moire", "label": "Moire"}],
                allow_back=False,
            )
        self.assertEqual(choice, "moire")

    def test_interactive_choice_returns_option_value_when_present(self):
        with mock.patch("builtins.input", return_value="2"), mock.patch("sys.stdout", new_callable=StringIO):
            choice = interactive_cli._choice(
                "Strain axis",
                [
                    {"key": "axis_a", "value": "a", "label": "a axis"},
                    {"key": "axis_b", "value": "b", "label": "b axis"},
                ],
            )
        self.assertEqual(choice, "b")

    def test_interactive_path_picker_accepts_back_shortcut(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            with mock.patch("builtins.input", return_value="b"), mock.patch("sys.stdout", new_callable=StringIO):
                with self.assertRaises(interactive_cli._BackInteractive):
                    interactive_cli._prompt_path(
                        "Choose a structure",
                        patterns=("*.vasp",),
                        roots=(temp_root,),
                    )
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_adsorption_site_preview_shows_direct_and_cartesian_coordinates(self):
        sites = [
            surface.AdsorptionSite("top", (0.25, 0.5, 0.75), (1.0, 2.0, 3.0)),
        ]
        preview = format_adsorption_sites(sites)
        self.assertIn("direct (u, v, w)", preview)
        self.assertIn("cartesian (x, y, z) Ang", preview)
        self.assertIn("top", preview)

    def test_defect_native_analysis_groups_fcc_bulk_sites(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            fcc_path = temp_root / "Au_fcc.vasp"
            io.write_poscar(
                str(fcc_path),
                np.eye(3) * 4.0,
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0],
                    ],
                    dtype=float,
                ),
                [4],
                ["Au"],
                comment="fcc bulk",
                positions_are_cartesian=False,
            )
            tool = DefectWorkflow(runs_root=str(temp_root / "runs"), output_root=str(temp_root / "output"))
            result = tool.analyse(str(fcc_path), structure_kind="bulk", backend="native")
            atom_sites = [site for site in result.payload["analysis"]["sites"] if site["site_kind"] == "atom"]
            self.assertEqual(len(atom_sites), 1)
            self.assertEqual(atom_sites[0]["multiplicity"], 4)
            self.assertIn("defect_analysis.json", result.artifacts["analysis_json"])
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_symmetry_native_analysis_writes_report_without_spglib(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            structure_path = temp_root / "Au_fcc.vasp"
            io.write_poscar(
                str(structure_path),
                np.eye(3) * 4.0,
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0],
                    ],
                    dtype=float,
                ),
                [4],
                ["Au"],
                comment="fcc bulk",
                positions_are_cartesian=False,
            )
            tool = SymmetryWorkflow(runs_root=str(temp_root / "runs"), output_root=str(temp_root / "output"))
            result = tool.analyse(str(structure_path), backend="native")
            self.assertEqual(result.summary["backend"], "native")
            self.assertEqual(result.payload["analysis"]["operation_count"], 0)
            self.assertIn("symmetry_analysis.json", result.artifacts["analysis_json"])
            self.assertIn("exact space group", result.payload["symmetry_preview"].lower())
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_symmetry_cli_analyse_native(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            structure_path = temp_root / "single.vasp"
            io.write_poscar(
                str(structure_path),
                np.eye(3) * 4.0,
                np.array([[0.0, 0.0, 0.0]], dtype=float),
                [1],
                ["X"],
                comment="single bulk",
                positions_are_cartesian=False,
            )
            namespace = moire_cli.build_parser().parse_args(["symmetry", "analyse", str(structure_path), "--backend", "native"])
            tool = SymmetryWorkflow(
                runs_root=str(temp_root / "runs"),
                output_root=str(temp_root / "output"),
            )
            with mock.patch.object(cli_main, "Symmetry", return_value=tool):
                result = cli_main.execute_namespace(namespace)
            self.assertIn("symmetry_preview", result.payload)
            self.assertEqual(result.summary["backend"], "native")
            self.assertTrue(Path(result.manifest_path).is_relative_to(temp_root))
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_symmetry_spglib_identifies_fcc_and_reduces_primitive(self):
        if find_spec("spglib") is None:
            self.skipTest("spglib is not installed")
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            structure_path = temp_root / "Au_fcc.vasp"
            io.write_poscar(
                str(structure_path),
                np.eye(3) * 4.0,
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0],
                    ],
                    dtype=float,
                ),
                [4],
                ["Au"],
                comment="fcc bulk",
                positions_are_cartesian=False,
            )
            tool = SymmetryWorkflow(runs_root=str(temp_root / "runs"), output_root=str(temp_root / "output"))
            analysis = tool.analyse(str(structure_path), backend="spglib")
            self.assertEqual(analysis.summary["space_group_number"], 225)
            groups = analysis.payload["analysis"]["equivalent_groups"]
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0]["multiplicity"], 4)

            reduced = tool.reduce(str(structure_path), cell="primitive", backend="spglib")
            primitive = io.read_poscar(reduced.artifacts["output_poscar"])
            self.assertEqual(primitive.natoms, 1)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_symmetry_cli_reduce_conventional_requires_spglib(self):
        if find_spec("spglib") is None:
            self.skipTest("spglib is not installed")
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            structure_path = temp_root / "single.vasp"
            io.write_poscar(
                str(structure_path),
                np.eye(3) * 4.0,
                np.array([[0.0, 0.0, 0.0]], dtype=float),
                [1],
                ["X"],
                comment="single bulk",
                positions_are_cartesian=False,
            )
            namespace = moire_cli.build_parser().parse_args(
                ["symmetry", "reduce", str(structure_path), "--cell", "conventional", "--backend", "spglib"]
            )
            tool = SymmetryWorkflow(
                runs_root=str(temp_root / "runs"),
                output_root=str(temp_root / "output"),
            )
            with mock.patch.object(cli_main, "Symmetry", return_value=tool):
                result = cli_main.execute_namespace(namespace)
            self.assertTrue(Path(result.artifacts["output_poscar"]).exists())
            self.assertEqual(result.summary["cell"], "conventional")
            self.assertTrue(Path(result.artifacts["output_poscar"]).is_relative_to(temp_root))
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_defect_vacancy_generation_preserves_selective_flags(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            structure_path = temp_root / "selective.vasp"
            io.write_poscar(
                str(structure_path),
                np.eye(3) * 4.0,
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0],
                    ],
                    dtype=float,
                ),
                [4],
                ["Au"],
                comment="selective fcc",
                positions_are_cartesian=False,
                selective_flags=[("F", "F", "F"), ("T", "T", "T"), ("T", "F", "T"), ("F", "T", "F")],
            )
            tool = DefectWorkflow(runs_root=str(temp_root / "runs"), output_root=str(temp_root / "output"))
            result = tool.generate(
                str(structure_path),
                "vacancy",
                structure_kind="bulk",
                backend="native",
                site_ids=["atom_001"],
            )
            generated = io.read_poscar(result.artifacts["structures"][0])
            self.assertEqual(generated.natoms, 3)
            self.assertEqual(generated.counts, [3])
            self.assertTrue(generated.selective_dynamics)
            self.assertEqual(len(generated.selective_flags), 3)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_defect_substitution_and_interstitial_generation_update_counts(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            structure_path = temp_root / "binary.vasp"
            io.write_poscar(
                str(structure_path),
                np.eye(3) * 5.0,
                np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=float),
                [1, 1],
                ["A", "B"],
                comment="binary bulk",
                positions_are_cartesian=False,
            )
            tool = DefectWorkflow(runs_root=str(temp_root / "runs"), output_root=str(temp_root / "output"))
            sub_result = tool.generate(
                str(structure_path),
                "substitution",
                structure_kind="bulk",
                backend="native",
                site_ids=["atom_001"],
                substitution_species="C",
            )
            substituted = io.read_poscar(sub_result.artifacts["structures"][0])
            self.assertEqual(substituted.natoms, 2)
            self.assertIn("C", substituted.species)

            int_result = tool.generate(
                str(structure_path),
                "interstitial",
                structure_kind="bulk",
                backend="native",
                site_ids=["interstitial_001"],
                species="H",
            )
            interstitial = io.read_poscar(int_result.artifacts["structures"][0])
            self.assertEqual(interstitial.natoms, 3)
            self.assertIn("H", interstitial.species)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_defect_preview_cli_prints_available_site_table(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            structure_path = temp_root / "single.vasp"
            io.write_poscar(
                str(structure_path),
                np.eye(3) * 4.0,
                np.array([[0.0, 0.0, 0.0]], dtype=float),
                [1],
                ["X"],
                comment="single bulk",
                positions_are_cartesian=False,
            )
            namespace = moire_cli.build_parser().parse_args(
                ["defect", "preview", str(structure_path), "--structure-kind", "bulk", "--backend", "native"]
            )
            tool = DefectWorkflow(
                runs_root=str(temp_root / "runs"),
                output_root=str(temp_root / "output"),
            )
            with mock.patch.object(cli_main, "Defect", return_value=tool):
                result = cli_main.execute_namespace(namespace)
            self.assertIn("defect_preview", result.payload)
            self.assertIn("atom_001", result.payload["defect_preview"])
            self.assertTrue(Path(result.manifest_path).is_relative_to(temp_root))
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_defect_adatom_generation_top_and_bottom(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            structure_path = temp_root / "slab.vasp"
            io.write_poscar(
                str(structure_path),
                np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 20.0]]),
                np.array([
                    [0.0, 0.0, 0.1],
                    [0.5, 0.5, 0.5]
                ], dtype=float),
                [2],
                ["Au"],
                comment="slab",
                positions_are_cartesian=False,
            )
            tool = DefectWorkflow(runs_root=str(temp_root / "runs"), output_root=str(temp_root / "output"))
            
            top_result = tool.generate(
                str(structure_path),
                "adatom",
                structure_kind="surface",
                backend="native",
                surface_side="top",
                species="H",
                height=2.5,
                site_ids=["adatom_top_001"],
            )
            self.assertEqual(len(top_result.artifacts["structures"]), 1)
            top_adatom_struct = io.read_poscar(top_result.artifacts["structures"][0])
            self.assertEqual(top_adatom_struct.natoms, 3)
            self.assertAlmostEqual(float(top_adatom_struct.positions_direct[-1, 2]), 0.625, places=3)
            
            bottom_result = tool.generate(
                str(structure_path),
                "adatom",
                structure_kind="surface",
                backend="native",
                surface_side="bottom",
                species="H",
                height=2.5,
                site_ids=["adatom_top_001"],
            )
            self.assertEqual(len(bottom_result.artifacts["structures"]), 1)
            bottom_adatom_struct = io.read_poscar(bottom_result.artifacts["structures"][0])
            self.assertEqual(bottom_adatom_struct.natoms, 3)
            self.assertAlmostEqual(float(bottom_adatom_struct.positions_direct[-1, 2]), -0.025, places=3)

        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_rotation_matrices_x_y(self):
        r_x = lattice.rotation_matrix_x(90.0)
        res = r_x @ np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(res[0], 0.0)
        self.assertAlmostEqual(res[1], 0.0)
        self.assertAlmostEqual(res[2], 1.0)

        r_y = lattice.rotation_matrix_y(90.0)
        res_y = r_y @ np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(res_y[0], 0.0)
        self.assertAlmostEqual(res_y[1], 0.0)
        self.assertAlmostEqual(res_y[2], -1.0)

        r_comb = lattice.yaw_pitch_roll_matrix(90.0, 90.0, 90.0)
        self.assertEqual(r_comb.shape, (3, 3))

    def test_adsorbate_place_with_tilt_and_roll(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            substrate_path = temp_root / "sub.vasp"
            io.write_poscar(
                str(substrate_path),
                np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 15.0]]),
                np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.2]], dtype=float),
                [2],
                ["Au"],
                comment="sub",
            )
            molecule_path = temp_root / "mol.vasp"
            io.write_poscar(
                str(molecule_path),
                np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]),
                np.array([[0.5, 0.5, 0.45], [0.5, 0.5, 0.55]], dtype=float),
                [2],
                ["H"],
                comment="diatomic",
            )
            tool = MoleculeWorkflow(runs_root=str(temp_root / "runs"), output_root=str(temp_root / "output"))
            res = tool.place(
                substrate_poscar=str(substrate_path),
                molecule_poscar=str(molecule_path),
                site_type="top",
                site_index=1,
                height=2.0,
                rotation_deg=45.0,
                tilt_deg=30.0,
                roll_deg=15.0,
            )
            output_poscar = io.read_poscar(res.artifacts["output_poscar"])
            self.assertEqual(output_poscar.natoms, 4)
            self.assertEqual(output_poscar.species, ["Au", "H"])
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_defect_divacancy_generation(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            structure_path = temp_root / "slab.vasp"
            io.write_poscar(
                str(structure_path),
                np.array([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]]),
                np.array([
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5]
                ], dtype=float),
                [4],
                ["Au"],
                comment="cubic",
            )
            tool = DefectWorkflow(runs_root=str(temp_root / "runs"), output_root=str(temp_root / "output"))
            analysis_res = tool.analyse(
                str(structure_path),
                divacancy_distance=5.0,
                backend="native",
            )
            analysis_dict = analysis_res.payload["analysis"]
            divacancy_sites = [s for s in analysis_dict["sites"] if s["site_kind"] == "divacancy"]
            self.assertTrue(len(divacancy_sites) > 0)
            
            gen_res = tool.generate(
                str(analysis_res.manifest_path),
                defect_type="divacancy",
                divacancy_distance=5.0,
            )
            self.assertEqual(len(gen_res.artifacts["structures"]), len(divacancy_sites))
            first_struct = io.read_poscar(gen_res.artifacts["structures"][0])
            self.assertEqual(first_struct.natoms, 2)
            
            manual_gen = tool.generate(
                str(analysis_res.manifest_path),
                defect_type="divacancy",
                site_ids=["atom_001", "atom_002"],
            )
            self.assertEqual(len(manual_gen.artifacts["structures"]), 1)
            manual_struct = io.read_poscar(manual_gen.artifacts["structures"][0])
            self.assertEqual(manual_struct.natoms, 2)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_defect_antisite_autoresolution(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            binary_path = temp_root / "gaas.vasp"
            io.write_poscar(
                str(binary_path),
                np.eye(3) * 5.0,
                np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=float),
                [1, 1],
                ["Ga", "As"],
                comment="gaas",
            )
            tool = DefectWorkflow(runs_root=str(temp_root / "runs"), output_root=str(temp_root / "output"))
            res = tool.generate(
                str(binary_path),
                defect_type="antisite",
                structure_kind="bulk",
            )
            self.assertEqual(len(res.artifacts["structures"]), 2)
            struct1 = io.read_poscar(res.artifacts["structures"][0])
            self.assertEqual(struct1.species, ["As"])
            self.assertEqual(list(struct1.counts), [2])

            struct2 = io.read_poscar(res.artifacts["structures"][1])
            self.assertEqual(struct2.species, ["Ga"])
            self.assertEqual(list(struct2.counts), [2])
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_interface_surface_and_build_wrappers_create_artifacts(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            bulk_path = temp_root / "bulk_simple.vasp"
            io.write_poscar(
                str(bulk_path),
                np.eye(3),
                np.array([[0.0, 0.0, 0.0]], dtype=float),
                [1],
                ["X"],
                comment="simple cubic bulk",
                positions_are_cartesian=False,
            )

            surface_tool = InterfaceSurface(runs_root=str(temp_root / "runs"), output_root=str(temp_root / "output"))
            bottom_result = surface_tool.surface(
                bulk_poscar=str(bulk_path),
                miller="1,0,0",
                layers=3,
                vacuum=8.0,
            )
            top_result = surface_tool.surface(
                bulk_poscar=str(bulk_path),
                miller="1,1,0",
                layers=3,
                vacuum=8.0,
            )
            self.assertTrue(Path(bottom_result.artifacts["slab_poscar"]).exists())
            self.assertTrue(Path(top_result.artifacts["slab_poscar"]).exists())

            interface_tool = InterfaceWorkflow(runs_root=str(temp_root / "runs"), output_root=str(temp_root / "output"))
            build_result = interface_tool.build(
                bottom_input=bottom_result.artifacts["slab_poscar"],
                top_input=top_result.artifacts["slab_poscar"],
                gap=3.0,
            )
            self.assertTrue(Path(build_result.artifacts["interface_poscar"]).exists())
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_surface_wrapper_uses_non_overwriting_descriptive_output_names(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            bulk_path = temp_root / "au_bulk.vasp"
            io.write_poscar(
                str(bulk_path),
                4.08 * np.eye(3),
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0],
                    ],
                    dtype=float,
                ),
                [4],
                ["Au"],
                comment="fcc gold bulk",
                positions_are_cartesian=False,
            )

            surface_tool = InterfaceSurface(runs_root=str(temp_root / "runs"), output_root=str(temp_root / "output"))
            first = surface_tool.surface(
                bulk_poscar=str(bulk_path),
                miller="111",
                layers=3,
                vacuum=12.0,
                analyse_sites=True,
            )
            second = surface_tool.surface(
                bulk_poscar=str(bulk_path),
                miller="111",
                layers=4,
                vacuum=12.0,
                supercell_matrix=(1, 1, 0, 2),
                analyse_sites=True,
            )

            first_slab = Path(first.artifacts["slab_poscar"])
            second_slab = Path(second.artifacts["slab_poscar"])
            self.assertNotEqual(first_slab, second_slab)
            self.assertTrue(first_slab.exists())
            self.assertTrue(second_slab.exists())
            self.assertIn("hkl111", first_slab.name)
            self.assertIn("L03", first_slab.name)
            self.assertIn("m1_1_0_2", second_slab.name)
            self.assertRegex(first_slab.stem, r"_\d{6}-\d{4}$")
            self.assertRegex(second_slab.stem, r"_\d{6}-\d{4}$")
            self.assertNotEqual(Path(first.artifacts["sites_json"]), Path(second.artifacts["sites_json"]))
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_surface_builder_creates_a_slab_from_a_simple_cubic_bulk(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            bulk_path = temp_root / "bulk_simple.vasp"
            io.write_poscar(
                str(bulk_path),
                np.eye(3),
                np.array([[0.0, 0.0, 0.0]], dtype=float),
                [1],
                ["X"],
                comment="simple cubic bulk",
                positions_are_cartesian=False,
            )

            run = surface.build_surface(
                str(bulk_path),
                miller=(1, 1, 0),
                layers=3,
                vacuum=8.0,
                output_path=str(temp_root / "surface_110.vasp"),
            )
            self.assertTrue(run.output_path.exists())
            self.assertEqual(run.total_atoms, 3)

            slab = io.read_poscar(str(run.output_path))
            self.assertEqual(slab.natoms, 3)
            self.assertGreater(float(np.linalg.norm(slab.lattice[2])), 2.0)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_surface_builder_vacuum_is_total_requested_empty_space(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            bulk_path = temp_root / "au_bulk.vasp"
            io.write_poscar(
                str(bulk_path),
                4.08 * np.eye(3),
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0],
                    ],
                    dtype=float,
                ),
                [4],
                ["Au"],
                comment="fcc gold bulk",
                positions_are_cartesian=False,
            )

            run = surface.build_surface(
                str(bulk_path),
                miller=(1, 1, 1),
                layers=4,
                vacuum=15.0,
                output_path=str(temp_root / "au111_vac15.vasp"),
            )
            slab = io.read_poscar(str(run.output_path))
            normal = surface._surface_normal(slab.lattice)
            projections = slab.positions_cartesian @ normal
            slab_thickness = float(projections.max() - projections.min())
            c_length = float(np.linalg.norm(slab.lattice[2]))
            levels = [center for center, _ in surface._cluster_projection_levels(projections, 0.35)]
            self.assertGreater(float(projections.min()), 0.0)
            self.assertAlmostEqual(float(projections.min()), float(levels[1] - levels[0]), places=6)
            self.assertAlmostEqual(c_length - slab_thickness, 15.0, places=6)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_surface_builder_scales_in_plane_with_supercell_matrix(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            bulk_path = temp_root / "bulk_simple.vasp"
            io.write_poscar(
                str(bulk_path),
                np.eye(3),
                np.array([[0.0, 0.0, 0.0]], dtype=float),
                [1],
                ["X"],
                comment="simple cubic bulk",
                positions_are_cartesian=False,
            )

            base_run = surface.build_surface(
                str(bulk_path),
                miller=(1, 0, 0),
                layers=3,
                vacuum=8.0,
                output_path=str(temp_root / "surface_base.vasp"),
            )
            scaled_run = surface.build_surface(
                str(bulk_path),
                miller=(1, 0, 0),
                layers=3,
                vacuum=8.0,
                supercell_matrix=(2, 0, 0, 3),
                output_path=str(temp_root / "surface_scaled.vasp"),
            )

            self.assertEqual(scaled_run.total_atoms, 6 * base_run.total_atoms)

            base_slab = io.read_poscar(str(base_run.output_path))
            scaled_slab = io.read_poscar(str(scaled_run.output_path))
            base_area = float(np.linalg.norm(np.cross(base_slab.lattice[0], base_slab.lattice[1])))
            scaled_area = float(np.linalg.norm(np.cross(scaled_slab.lattice[0], scaled_slab.lattice[1])))
            self.assertAlmostEqual(scaled_area, 6.0 * base_area, places=8)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_surface_builder_accepts_non_diagonal_supercell_matrix(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            bulk_path = temp_root / "au_bulk.vasp"
            io.write_poscar(
                str(bulk_path),
                4.08 * np.eye(3),
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0],
                    ],
                    dtype=float,
                ),
                [4],
                ["Au"],
                comment="fcc gold bulk",
                positions_are_cartesian=False,
            )

            base_run = surface.build_surface(
                str(bulk_path),
                miller=(1, 1, 1),
                layers=4,
                vacuum=12.0,
                output_path=str(temp_root / "au111_base.vasp"),
            )
            sheared_run = surface.build_surface(
                str(bulk_path),
                miller=(1, 1, 1),
                layers=4,
                vacuum=12.0,
                supercell_matrix=(1, 1, 0, 2),
                output_path=str(temp_root / "au111_sheared.vasp"),
            )

            self.assertEqual(sheared_run.supercell_matrix, (1, 1, 0, 2))
            self.assertEqual(sheared_run.total_atoms, 2 * base_run.total_atoms)
            base_slab = io.read_poscar(str(base_run.output_path))
            sheared_slab = io.read_poscar(str(sheared_run.output_path))
            base_area = float(np.linalg.norm(np.cross(base_slab.lattice[0], base_slab.lattice[1])))
            sheared_area = float(np.linalg.norm(np.cross(sheared_slab.lattice[0], sheared_slab.lattice[1])))
            self.assertAlmostEqual(sheared_area, 2.0 * base_area, places=8)
            self.assertEqual(_stacking_sequence(sheared_slab), "ABCA")
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_surface_builder_reports_top_bridge_hcp_and_fcc_sites_for_fcc_111(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            bulk_path = temp_root / "au_bulk.vasp"
            lattice_bulk = np.array(
                [
                    [4.08, 0.0, 0.0],
                    [0.0, 4.08, 0.0],
                    [0.0, 0.0, 4.08],
                ],
                dtype=float,
            )
            positions_bulk = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.5],
                    [0.5, 0.0, 0.5],
                    [0.5, 0.5, 0.0],
                ],
                dtype=float,
            )
            io.write_poscar(
                str(bulk_path),
                lattice_bulk,
                positions_bulk,
                [4],
                ["Au"],
                comment="fcc gold bulk",
                positions_are_cartesian=False,
            )

            run = surface.build_surface(
                str(bulk_path),
                miller=(1, 1, 1),
                layers=4,
                vacuum=12.0,
                analyse_sites=True,
                output_path=str(temp_root / "au111.vasp"),
            )
            self.assertTrue(run.output_path.exists())
            self.assertIsNotNone(run.site_output_path)
            self.assertEqual(run.total_atoms, 4)
            self.assertEqual(run.site_counts.get("top"), 1)
            self.assertEqual(run.site_counts.get("bridge"), 3)
            self.assertEqual(run.site_counts.get("hcp_hollow"), 1)
            self.assertEqual(run.site_counts.get("fcc_hollow"), 1)

            slab = io.read_poscar(str(run.output_path))
            self.assertEqual(_stacking_sequence(slab), "ABCA")
            report = surface.find_adsorption_sites(slab)
            self.assertAlmostEqual(report.average_top_layer_coordination, 6.0, places=6)
            self.assertEqual(report.site_counts.get("top"), 1)
            self.assertEqual(report.site_counts.get("bridge"), 3)
            self.assertEqual(report.site_counts.get("hcp_hollow"), 1)
            self.assertEqual(report.site_counts.get("fcc_hollow"), 1)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_surface_builder_prefers_obtuse_primitive_cell_for_fcc_111(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            bulk_path = temp_root / "au_bulk.vasp"
            io.write_poscar(
                str(bulk_path),
                4.08 * np.eye(3),
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0],
                    ],
                    dtype=float,
                ),
                [4],
                ["Au"],
                comment="fcc gold bulk",
                positions_are_cartesian=False,
            )

            analysis = surface.analyse_primitive_surface(str(bulk_path), miller=(1, 1, 1), probe_layers=6)
            self.assertEqual(analysis.stacking_period, "ABC")
            self.assertAlmostEqual(analysis.inplane_angle_deg, 120.0, places=6)

            base_run = surface.build_surface(
                str(bulk_path),
                miller=(1, 1, 1),
                layers=4,
                vacuum=12.0,
                output_path=str(temp_root / "au111.vasp"),
            )
            scaled_run = surface.build_surface(
                str(bulk_path),
                miller=(1, 1, 1),
                layers=4,
                vacuum=12.0,
                supercell_matrix=(2, 0, 0, 3),
                output_path=str(temp_root / "au111_scaled.vasp"),
            )
            self.assertEqual(base_run.total_atoms, 4)
            self.assertEqual(scaled_run.total_atoms, 24)

            scaled = io.read_poscar(str(scaled_run.output_path))
            angle = float(
                np.degrees(
                    np.arccos(
                        np.clip(
                            float(np.dot(scaled.lattice[0], scaled.lattice[1]))
                            / max(float(np.linalg.norm(scaled.lattice[0]) * np.linalg.norm(scaled.lattice[1])), 1e-12),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
            self.assertAlmostEqual(angle, 120.0, places=6)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_surface_builder_reports_abab_stacking_for_fcc_001(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            bulk_path = temp_root / "au_bulk.vasp"
            io.write_poscar(
                str(bulk_path),
                4.08 * np.eye(3),
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0],
                    ],
                    dtype=float,
                ),
                [4],
                ["Au"],
                comment="fcc gold bulk",
                positions_are_cartesian=False,
            )

            run = surface.build_surface(
                str(bulk_path),
                miller=(0, 0, 1),
                layers=4,
                vacuum=12.0,
                output_path=str(temp_root / "au001.vasp"),
            )
            self.assertTrue(run.output_path.exists())
            self.assertEqual(run.total_atoms, 4)
            slab = io.read_poscar(str(run.output_path))
            self.assertEqual(_stacking_sequence(slab), "ABAB")
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_interactive_adsorbate_site_options_only_include_detected_families(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            bulk_path = temp_root / "au_bulk.vasp"
            io.write_poscar(
                str(bulk_path),
                4.08 * np.eye(3),
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0],
                    ],
                    dtype=float,
                ),
                [4],
                ["Au"],
                comment="fcc gold bulk",
                positions_are_cartesian=False,
            )
            slab_run = surface.build_surface(
                str(bulk_path),
                miller=(1, 0, 0),
                layers=4,
                vacuum=12.0,
                output_path=str(temp_root / "au100.vasp"),
            )
            report = surface.find_adsorption_sites(str(slab_run.output_path))
            keys = {option["key"] for option in interactive_cli._site_options_from_report(report)}

            self.assertEqual(keys, {"top", "bridge", "fourfold_hollow"})
            self.assertNotIn("fcc_hollow", keys)
            self.assertNotIn("hcp_hollow", keys)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_surface_builder_reduces_body_centred_cells_before_slab_generation(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            bulk_path = temp_root / "fe_bcc.vasp"
            io.write_poscar(
                str(bulk_path),
                2.87 * np.eye(3),
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.5, 0.5, 0.5],
                    ],
                    dtype=float,
                ),
                [2],
                ["Fe"],
                comment="bcc iron bulk",
                positions_are_cartesian=False,
            )

            run = surface.build_surface(
                str(bulk_path),
                miller=(0, 0, 1),
                layers=4,
                vacuum=10.0,
                output_path=str(temp_root / "fe001.vasp"),
            )
            self.assertEqual(run.total_atoms, 4)
            slab = io.read_poscar(str(run.output_path))
            self.assertEqual(_stacking_sequence(slab), "ABAB")
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_adsorb_places_molecule_on_selected_site_with_requested_height(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            bulk_path = temp_root / "au_bulk.vasp"
            lattice_bulk = np.array(
                [
                    [4.08, 0.0, 0.0],
                    [0.0, 4.08, 0.0],
                    [0.0, 0.0, 4.08],
                ],
                dtype=float,
            )
            positions_bulk = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.5],
                    [0.5, 0.0, 0.5],
                    [0.5, 0.5, 0.0],
                ],
                dtype=float,
            )
            io.write_poscar(
                str(bulk_path),
                lattice_bulk,
                positions_bulk,
                [4],
                ["Au"],
                comment="fcc gold bulk",
                positions_are_cartesian=False,
            )
            molecule_path = temp_root / "co.vasp"
            io.write_poscar(
                str(molecule_path),
                10.0 * np.eye(3),
                np.array(
                    [
                        [0.5, 0.5, 0.5],
                        [0.5, 0.5, 0.613],
                    ],
                    dtype=float,
                ),
                [1, 1],
                ["C", "O"],
                comment="co molecule",
                positions_are_cartesian=False,
            )

            slab_run = surface.build_surface(
                str(bulk_path),
                miller=(1, 1, 1),
                layers=4,
                vacuum=12.0,
                output_path=str(temp_root / "au111.vasp"),
            )
            adsorb_run = molecule.place_molecule_on_site(
                str(slab_run.output_path),
                str(molecule_path),
                site_type="fcc",
                site_index=1,
                height=2.3,
                reframe_axes="none",
                output_path=str(temp_root / "au111_co.vasp"),
            )

            self.assertEqual(adsorb_run.site_type, "fcc_hollow")
            combined = io.read_poscar(str(adsorb_run.output_path))
            expanded_species = []
            for symbol, count in zip(combined.species, combined.counts):
                expanded_species.extend([symbol] * int(count))
            molecule_mask = np.array([symbol in {"C", "O"} for symbol in expanded_species], dtype=bool)
            self.assertEqual(int(np.count_nonzero(molecule_mask)), 2)

            molecule_positions = combined.positions_cartesian[molecule_mask]
            molecule_species = [expanded_species[index] for index in np.flatnonzero(molecule_mask).tolist()]
            molecule_com = molecule.center_of_mass_cartesian(molecule_positions, molecule_species)
            slab_normal = surface._surface_normal(combined.lattice)

            site_inplane = np.asarray(adsorb_run.site_cartesian, dtype=float) - float(np.dot(adsorb_run.site_cartesian, slab_normal)) * slab_normal
            com_inplane = molecule_com - float(np.dot(molecule_com, slab_normal)) * slab_normal
            self.assertTrue(np.allclose(com_inplane, site_inplane, atol=1e-6))

            lowest_projection = float(np.min(molecule_positions @ slab_normal))
            site_projection = float(np.dot(adsorb_run.site_cartesian, slab_normal))
            self.assertAlmostEqual(lowest_projection - site_projection, 2.3, places=6)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_adsorb_rejects_molecule_larger_than_substrate_cell(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            bulk_path = temp_root / "au_bulk.vasp"
            io.write_poscar(
                str(bulk_path),
                4.08 * np.eye(3),
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0],
                    ],
                    dtype=float,
                ),
                [4],
                ["Au"],
                comment="fcc gold bulk",
                positions_are_cartesian=False,
            )
            slab_run = surface.build_surface(
                str(bulk_path),
                miller=(1, 0, 0),
                layers=4,
                vacuum=12.0,
                output_path=str(temp_root / "au100.vasp"),
            )
            molecule_path = temp_root / "large_molecule.vasp"
            io.write_poscar(
                str(molecule_path),
                20.0 * np.eye(3),
                np.array(
                    [
                        [0.0, 0.0, 8.0],
                        [0.0, 9.0, 8.0],
                    ],
                    dtype=float,
                ),
                [2],
                ["C"],
                comment="too large molecule",
                positions_are_cartesian=True,
            )

            with self.assertRaisesRegex(ValueError, "cannot be contained in one periodic image"):
                molecule.place_molecule_on_site(
                    str(slab_run.output_path),
                    str(molecule_path),
                    site_type="top",
                    site_index=1,
                    height=2.0,
                )
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_adsorbate_workflow_can_enlarge_primitive_substrate_for_large_molecule(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            bulk_path = temp_root / "au_bulk.vasp"
            io.write_poscar(
                str(bulk_path),
                4.08 * np.eye(3),
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.5, 0.5],
                        [0.5, 0.0, 0.5],
                        [0.5, 0.5, 0.0],
                    ],
                    dtype=float,
                ),
                [4],
                ["Au"],
                comment="fcc gold bulk",
                positions_are_cartesian=False,
            )
            slab_run = surface.build_surface(
                str(bulk_path),
                miller=(1, 0, 0),
                layers=4,
                vacuum=12.0,
                output_path=str(temp_root / "au100_primitive.vasp"),
            )
            molecule_path = temp_root / "large_molecule.vasp"
            io.write_poscar(
                str(molecule_path),
                20.0 * np.eye(3),
                np.array(
                    [
                        [0.0, 0.0, 8.0],
                        [0.0, 9.0, 8.0],
                    ],
                    dtype=float,
                ),
                [2],
                ["C"],
                comment="large molecule",
                positions_are_cartesian=True,
            )

            tool = MoleculeWorkflow(runs_root=str(temp_root / "runs"), output_root=str(temp_root / "output"))
            result = tool.place(
                substrate_poscar=str(slab_run.output_path),
                molecule_poscar=str(molecule_path),
                site_type="top",
                site_index=1,
                height=2.0,
                auto_repeat_substrate=True,
            )

            self.assertTrue(Path(result.artifacts["output_poscar"]).exists())
            self.assertGreater(result.summary["substrate_atom_count"], slab_run.total_atoms)
            combined = io.read_poscar(str(result.artifacts["output_poscar"]))
            expanded_species = []
            for symbol, count in zip(combined.species, combined.counts):
                expanded_species.extend([symbol] * int(count))
            self.assertEqual(expanded_species.count("C"), 2)
            self.assertEqual(combined.natoms, result.summary["substrate_atom_count"] + 2)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_molecule_stage_isolates_and_rigidly_rotates_adsorbate(self):
        lattice_out, positions_direct, counts, species, flags = _simple_adsorbate_stack()

        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            stack_path = temp_root / "stacked_adsorbate.vasp"
            io.write_poscar(
                str(stack_path),
                lattice_out,
                positions_direct,
                counts,
                species,
                comment="stacked adsorbate test",
                positions_are_cartesian=False,
                selective_flags=flags,
            )

            stacked = io.read_poscar(str(stack_path))
            selection = molecule.identify_top_group(stacked)
            self.assertEqual(selection.molecule_atom_count, 2)
            self.assertEqual(selection.substrate_atom_count, 2)

            reference_index = selection.molecule_indices[0]
            before_vector = stacked.positions_cartesian[reference_index] - selection.center_of_mass_cartesian

            output_path = temp_root / "stacked_adsorbate_adjusted.vasp"
            run = molecule.transform_top_molecule(
                str(stack_path),
                output_path=str(output_path),
                target_direct=(0.5, 0.5),
                rotation_deg=90.0,
            )

            transformed = io.read_poscar(str(output_path))
            selection_after = molecule.identify_top_group(transformed, z_cutoff=run.z_cutoff)
            after_vector = transformed.positions_cartesian[reference_index] - selection_after.center_of_mass_cartesian

            self.assertTrue(np.allclose(selection_after.center_of_mass_cartesian, run.target_cartesian, atol=1e-6))

            expected_vector = before_vector @ lattice.rotation_matrix_z(90.0).T
            self.assertTrue(np.allclose(after_vector, expected_vector, atol=1e-6))
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_identify_top_group_respects_manual_cutoff_and_reports_matching_metadata(self):
        lattice_out, positions_direct, counts, species, flags = _simple_adsorbate_stack()

        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            stack_path = temp_root / "stacked_cutoff.vasp"
            io.write_poscar(
                str(stack_path),
                lattice_out,
                positions_direct,
                counts,
                species,
                comment="stacked cutoff test",
                positions_are_cartesian=False,
                selective_flags=flags,
            )

            stacked = io.read_poscar(str(stack_path))
            selection = molecule.identify_top_group(stacked)
            z_values = stacked.positions_cartesian[:, 2]

            self.assertEqual(int(np.count_nonzero(z_values > selection.z_cutoff)), selection.molecule_atom_count)
            self.assertAlmostEqual(selection.gap_size, 3.35, places=6)

            manual_cutoff = float(np.min(z_values[list(selection.molecule_indices)]) + 0.1)
            manual_selection = molecule.identify_top_group(stacked, z_cutoff=manual_cutoff)
            expected_manual_count = int(np.count_nonzero(z_values > manual_cutoff))

            self.assertEqual(manual_selection.molecule_atom_count, expected_manual_count)
            self.assertNotEqual(manual_selection.molecule_atom_count, selection.molecule_atom_count)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_molecule_reframe_handles_boundary_crossing_adsorbate(self):
        lattice_out, positions_direct, counts, species, flags = _simple_adsorbate_stack()

        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            stack_path = temp_root / "stacked_reframe.vasp"
            io.write_poscar(
                str(stack_path),
                lattice_out,
                positions_direct,
                counts,
                species,
                comment="stacked reframe test",
                positions_are_cartesian=False,
                selective_flags=flags,
            )

            output_path = temp_root / "stacked_reframed.vasp"
            run = molecule.transform_top_molecule(
                str(stack_path),
                output_path=str(output_path),
                target_direct=(0.5, 0.5),
                rotation_deg=90.0,
                reframe_axes="xy",
            )

            reframed = io.read_poscar(str(output_path))
            selection = molecule.identify_top_group(reframed, z_cutoff=run.z_cutoff)
            molecule_direct = reframed.positions_direct[np.array(selection.molecule_indices, dtype=int)]

            self.assertEqual(selection.molecule_atom_count, 2)
            self.assertTrue(np.allclose(selection.center_of_mass_cartesian, run.center_of_mass_after, atol=1e-6))
            self.assertTrue(np.allclose(selection.center_of_mass_cartesian, run.target_cartesian, atol=1e-6))
            self.assertLessEqual(float(np.max(molecule_direct[:, 0]) - np.min(molecule_direct[:, 0])), 1.0 + 1e-8)
            self.assertLessEqual(float(np.max(molecule_direct[:, 1]) - np.min(molecule_direct[:, 1])), 1.0 + 1e-8)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_layer_stage_shifts_only_the_upper_group(self):
        lattice_out, positions_direct, counts, species, flags = _simple_adsorbate_stack()

        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            stack_path = temp_root / "stacked_layer.vasp"
            io.write_poscar(
                str(stack_path),
                lattice_out,
                positions_direct,
                counts,
                species,
                comment="stacked layer test",
                positions_are_cartesian=False,
                selective_flags=flags,
            )

            before = io.read_poscar(str(stack_path))
            selection = molecule.identify_top_group(before)
            top_index = selection.molecule_indices[0]
            bottom_index = selection.substrate_indices[0]

            run = molecule.shift_top_layer(
                str(stack_path),
                output_path=str(temp_root / "stacked_layer_shifted.vasp"),
                shift_direct=(0.25, 0.125),
            )

            after = io.read_poscar(str(run.output_path))
            expected_top = before.positions_direct[top_index] + np.array([0.25, 0.125, 0.0])
            self.assertTrue(np.allclose(after.positions_direct[top_index], expected_top, atol=1e-8))
            self.assertTrue(np.allclose(after.positions_direct[bottom_index], before.positions_direct[bottom_index], atol=1e-8))
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_visualize_marker_sizes_follow_atomic_radius(self):
        self.assertGreater(_marker_size("Au", projection="2d"), _marker_size("C", projection="2d"))
        self.assertGreater(_marker_size("C", projection="2d"), _marker_size("H", projection="2d"))
        self.assertGreater(_marker_size("Au", projection="3d"), _marker_size("C", projection="3d"))


if __name__ == '__main__':
    unittest.main()
