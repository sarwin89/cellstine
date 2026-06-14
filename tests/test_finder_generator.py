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
from cellstine.cli import interactive as interactive_cli
from cellstine.cli import main as cli_main
from cellstine.core.previews import format_adsorption_sites, format_bilayer_candidates
from cellstine.defect.defect import Defect as DefectWorkflow
from cellstine.io import native as io
from cellstine.moire import finder as cell_finder
from cellstine.moire import generator as cell_generator
from cellstine.moire import angles, find, finder, findn, generator, lattice, make, maken
from cellstine.adsorbate import molecule
from cellstine.adsorbate.molecule import Molecule as MoleculeWorkflow
from cellstine.interface import surface_backend as surface
from cellstine.interface.surface import Surface as InterfaceSurface, _stacking_sequence
from cellstine.interface.interface import Interface as InterfaceWorkflow, parse_miller_notation
from cellstine.visualize.matplotlib_backend import _marker_size
from cellstine.visualize import results_plotly as visualize
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
GRAPHENE_PATH = _sample_path('graph.vasp')
CU110_PATH = _sample_path('Cu110_truncated_bulk.vasp')
RELAXED_OVERLAYER_PATH = _sample_path('relaxed_overlayer_monoclinic.vasp')
REFERENCE_FILES = {
    13.15: str(BASE_DIR / 'Results' / 'spc_POSCAR_13-B.vasp'),
    21.787: str(BASE_DIR / 'Results' / 'spc_POSCAR_21-B.vasp'),
    27.9: str(BASE_DIR / 'Results' / 'spc_POSCAR_27-B.vasp'),
}


class MoireToolkitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mos2 = io.read_poscar(MOS2_PATH)
        cls.graphene = io.read_poscar(GRAPHENE_PATH)
        cls.cu110 = io.read_poscar(CU110_PATH)
        cls.relaxed_overlayer = io.read_poscar(RELAXED_OVERLAYER_PATH)

    def test_symmetry_lcm(self):
        sym_mos2, sym_graphene, sym_lcm = lattice.combined_symmetry_limit(self.mos2.lattice, self.graphene.lattice)
        self.assertEqual(sym_mos2, 60)
        self.assertEqual(sym_graphene, 60)
        self.assertEqual(sym_lcm, 60)

    def test_angle_shortlist_contains_reference_family(self):
        shortlist = angles.find_commensurate_angles(
            self.mos2.lattice,
            self.mos2.lattice,
            nindex=12,
            strain_tolerance=2e-3,
            min_angle=0.0,
            max_angle=30.0,
        )
        found_angles = np.array([item.angle_deg for item in shortlist])
        self.assertTrue(np.any(np.isclose(found_angles, 13.1735, atol=0.03)))
        self.assertTrue(np.any(np.isclose(found_angles, 21.7868, atol=0.03)))
        self.assertTrue(np.any(np.isclose(found_angles, 27.7958, atol=0.03)))

    def test_graphene_angle_shortlist_reaches_symmetry_limit(self):
        symmetry_top, symmetry_bottom, symmetry_lcm = lattice.combined_symmetry_limit(
            self.graphene.lattice,
            self.graphene.lattice,
        )
        shortlist = angles.find_commensurate_angles(
            self.graphene.lattice,
            self.graphene.lattice,
            nindex=8,
            strain_tolerance=2e-3,
            min_angle=0.0,
            max_angle=symmetry_lcm,
        )
        found_angles = np.array([item.angle_deg for item in shortlist])
        self.assertEqual((symmetry_top, symmetry_bottom, symmetry_lcm), (60, 60, 60))
        self.assertTrue(np.any(np.isclose(found_angles, 60.0, atol=1e-6)))
        self.assertLessEqual(float(found_angles.max()), 60.0)

    def test_graphene_mos2_search_window_uses_lcm_limit(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        find_run = find.run_find(
            top_poscar=GRAPHENE_PATH,
            bottom_poscar=MOS2_PATH,
            top_lattice=self.graphene.lattice,
            bottom_lattice=self.mos2.lattice,
            top_atoms=self.graphene.natoms,
            bottom_atoms=self.mos2.natoms,
            nindex=8,
            vector_tolerance=2e-3,
            vector_strain_tolerance=2e-3,
            candidate_tolerance=2e-3,
            max_atoms=400,
            output_root=str(temp_root),
        )
        try:
            self.assertEqual(find_run.symmetry_lcm, 60)
            self.assertAlmostEqual(find_run.search_min_angle, 0.0, places=8)
            self.assertAlmostEqual(find_run.search_max_angle, 60.0, places=8)
            self.assertIn(0.0, find_run.angle_values)
            self.assertIn(60.0, find_run.angle_values)
            self.assertTrue(find_run.dat_path.exists())
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_reference_angles_atom_counts(self):
        expected_atoms = {13.15: 114, 21.787: 42, 27.9: 78}
        for angle, atom_count in expected_atoms.items():
            with self.subTest(angle=angle):
                results = finder.find_supercells(
                    self.mos2.lattice,
                    self.mos2.lattice,
                    None,
                    None,
                    angles=[angle],
                    nindex=12,
                    tol=2e-3,
                    lin_tol=2e-3,
                    atom_count1=self.mos2.natoms,
                    atom_count2=self.mos2.natoms,
                    max_atoms=200,
                    vector_strain_tol=2e-3,
                )
                self.assertTrue(results)
                self.assertEqual(results[0].total_atoms, atom_count)

    def test_two_stage_find_and_make(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            find_run = find.run_find(
                top_poscar=MOS2_PATH,
                bottom_poscar=MOS2_PATH,
                top_lattice=self.mos2.lattice,
                bottom_lattice=self.mos2.lattice,
                top_atoms=self.mos2.natoms,
                bottom_atoms=self.mos2.natoms,
                nindex=8,
                explicit_angles=[13.15, 21.787, 27.9],
                vector_tolerance=2e-3,
                vector_strain_tolerance=2e-3,
                candidate_tolerance=2e-3,
                max_atoms=300,
                output_root=str(temp_root),
            )
            self.assertTrue(find_run.dat_path.exists())
            self.assertGreater(len(find_run.candidates), 0)

            make_run = make.generate_from_results(
                str(find_run.dat_path),
                index=1,
                interlayer_distance=3.35,
                output_dir=str(temp_root),
            )
            self.assertTrue(make_run.output_path.exists())
            self.assertGreater(make_run.total_atoms, 0)

            batch_runs = make.generate_many_from_results(
                str(find_run.dat_path),
                indexes=[1, 2],
                interlayer_distance=3.35,
                output_dir=str(temp_root),
            )
            self.assertEqual(len(batch_runs), 2)
            self.assertTrue(all(run.output_path.exists() for run in batch_runs))
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_repeat_structure_along_c_scales_cell_and_atom_counts(self):
        repeated = io.repeat_structure_along_c(self.mos2, 3)

        self.assertEqual(repeated.natoms, 3 * self.mos2.natoms)
        self.assertEqual(repeated.counts, [3 * count for count in self.mos2.counts])
        self.assertAlmostEqual(
            float(np.linalg.norm(repeated.lattice[2])),
            3.0 * float(np.linalg.norm(self.mos2.lattice[2])),
            places=8,
        )
        self.assertTrue(np.all(repeated.positions_direct[:, 2] >= -1e-10))
        self.assertTrue(np.all(repeated.positions_direct[:, 2] <= 1.0 + 1e-10))

    def test_parallel_finder_matches_serial_results_for_explicit_angles(self):
        serial = finder.find_supercells(
            self.mos2.lattice,
            self.mos2.lattice,
            None,
            None,
            angles=[13.15, 21.787],
            nindex=12,
            tol=2e-3,
            lin_tol=2e-3,
            atom_count1=self.mos2.natoms,
            atom_count2=self.mos2.natoms,
            max_atoms=200,
            vector_strain_tol=2e-3,
            workers=1,
        )
        parallel = finder.find_supercells(
            self.mos2.lattice,
            self.mos2.lattice,
            None,
            None,
            angles=[13.15, 21.787],
            nindex=12,
            tol=2e-3,
            lin_tol=2e-3,
            atom_count1=self.mos2.natoms,
            atom_count2=self.mos2.natoms,
            max_atoms=200,
            vector_strain_tol=2e-3,
            workers=2,
        )

        self.assertEqual(len(serial), len(parallel))
        self.assertEqual(
            [finder.candidate_to_dict(item) for item in serial[:5]],
            [finder.candidate_to_dict(item) for item in parallel[:5]],
        )

    def test_src_finder_matches_reference_finder_for_explicit_angles(self):
        reference = finder.find_supercells(
            self.mos2.lattice,
            self.mos2.lattice,
            None,
            None,
            angles=[13.15, 21.787],
            nindex=12,
            tol=2e-3,
            lin_tol=2e-3,
            atom_count1=self.mos2.natoms,
            atom_count2=self.mos2.natoms,
            max_atoms=200,
            vector_strain_tol=2e-3,
            workers=1,
        )
        package_native = cell_finder.find_supercells(
            self.mos2.lattice,
            self.mos2.lattice,
            None,
            None,
            angles=[13.15, 21.787],
            nindex=12,
            tol=2e-3,
            lin_tol=2e-3,
            atom_count1=self.mos2.natoms,
            atom_count2=self.mos2.natoms,
            max_atoms=200,
            vector_strain_tol=2e-3,
            workers=1,
        )
        self.assertEqual(
            [finder.candidate_to_dict(item) for item in reference[:5]],
            [cell_finder.candidate_to_dict(item) for item in package_native[:5]],
        )

    def test_make_matches_reference_counts(self):
        if not all(Path(path).exists() for path in REFERENCE_FILES.values()):
            self.skipTest("reference POSCAR files are not present in Results/")
        for angle, reference_path in REFERENCE_FILES.items():
            with self.subTest(angle=angle):
                results = finder.find_supercells(
                    self.mos2.lattice,
                    self.mos2.lattice,
                    None,
                    None,
                    angles=[angle],
                    nindex=12,
                    tol=2e-3,
                    lin_tol=2e-3,
                    atom_count1=self.mos2.natoms,
                    atom_count2=self.mos2.natoms,
                    max_atoms=200,
                    vector_strain_tol=2e-3,
                )
                best = results[0]
                temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
                temp_root.mkdir(parents=True, exist_ok=False)
                try:
                    dat_path = temp_root / "find_results.dat"
                    finder.write_results_dat(
                        str(dat_path),
                        MOS2_PATH,
                        MOS2_PATH,
                        [best],
                        run_id="test_reference",
                        parameters={"test_case": angle},
                    )
                    run = make.generate_from_results(
                        str(dat_path),
                        index=1,
                        interlayer_distance=0.0,
                        output_dir=str(temp_root),
                    )
                    generated = io.read_poscar(str(run.output_path))
                    reference = io.read_poscar(reference_path)
                    self.assertEqual(generated.counts, reference.counts)
                finally:
                    shutil.rmtree(temp_root, ignore_errors=True)

    def test_src_generator_matches_reference_supercell_build(self):
        results = finder.find_supercells(
            self.mos2.lattice,
            self.mos2.lattice,
            None,
            None,
            angles=[13.15],
            nindex=12,
            tol=2e-3,
            lin_tol=2e-3,
            atom_count1=self.mos2.natoms,
            atom_count2=self.mos2.natoms,
            max_atoms=200,
            vector_strain_tol=2e-3,
        )
        best = results[0]
        candidate_dict = finder.candidate_to_dict(best, 1)
        record = cell_generator.record_from_candidate_dict(candidate_dict, 1)

        reference_output = generator.build_supercell(
            MOS2_PATH,
            MOS2_PATH,
            record,
            interlayer_distance=3.35,
            preserve_layer="2",
            tolerance=1,
            tolerance_float=1e-4,
        )
        package_native_output = cell_generator.build_supercell(
            MOS2_PATH,
            MOS2_PATH,
            record,
            interlayer_distance=3.35,
            preserve_layer="2",
            tolerance=1,
            tolerance_float=1e-4,
        )

        self.assertTrue(np.allclose(reference_output[0], package_native_output[0]))
        self.assertTrue(np.allclose(reference_output[1], package_native_output[1]))
        self.assertEqual(reference_output[2], package_native_output[2])
        self.assertEqual(reference_output[3], package_native_output[3])

    def test_matrix_value_helper_matches_any_order(self):
        candidate = lattice.SupercellCandidate(
            angle_deg=43.3139,
            strain_avg=0.0,
            strain_layer1=0.0,
            strain_layer2=0.0,
            ratio1=1,
            ratio2=11,
            total_atoms=100,
            layer1_vector1=(-1, 0),
            layer1_vector2=(0, -1),
            layer2_vector1=(-3, -2),
            layer2_vector2=(4, -1),
            eps1=0.0,
            eps2=0.0,
            vector_product=1.0,
            area1=1.0,
            area2=11.0,
        )

        self.assertTrue(
            finder.candidate_matches_matrix_values(
                candidate,
                [1, 0, 1, 0],
                matrix_layer="1",
                matrix_match_mode="absolute",
            )
        )
        self.assertTrue(
            finder.candidate_matches_matrix_values(
                candidate,
                [-1, 0, 0, -1],
                matrix_layer="1",
                matrix_match_mode="exact",
            )
        )
        self.assertFalse(
            finder.candidate_matches_matrix_values(
                candidate,
                [1, 0, 1, 0],
                matrix_layer="1",
                matrix_match_mode="exact",
            )
        )
        self.assertTrue(
            finder.candidate_matches_matrix_values(
                candidate,
                [2, 1, 3, 4],
                matrix_layer="2",
                matrix_match_mode="absolute",
            )
        )

    def test_matrix_value_filter_can_keep_or_remove_candidates(self):
        filtered = finder.find_supercells(
            self.mos2.lattice,
            self.mos2.lattice,
            None,
            None,
            angles=[0.0],
            nindex=2,
            tol=2e-3,
            lin_tol=2e-3,
            atom_count1=self.mos2.natoms,
            atom_count2=self.mos2.natoms,
            max_atoms=100,
            vector_strain_tol=2e-3,
            matrix_values=[0, 0, 1, 1],
            matrix_layer="either",
            matrix_match_mode="absolute",
        )
        self.assertTrue(filtered)
        self.assertTrue(
            all(
                finder.candidate_matches_matrix_values(
                    candidate,
                    [0, 0, 1, 1],
                    matrix_layer="either",
                    matrix_match_mode="absolute",
                )
                for candidate in filtered
            )
        )

        impossible = finder.find_supercells(
            self.mos2.lattice,
            self.mos2.lattice,
            None,
            None,
            angles=[0.0],
            nindex=2,
            tol=2e-3,
            lin_tol=2e-3,
            atom_count1=self.mos2.natoms,
            atom_count2=self.mos2.natoms,
            max_atoms=100,
            vector_strain_tol=2e-3,
            matrix_values=[9, 9, 9, 9],
            matrix_layer="either",
            matrix_match_mode="absolute",
        )
        self.assertEqual(impossible, [])

    def test_canonicalized_degenerate_search_matches_full_dedup(self):
        full_candidates = finder.find_supercells(
            self.mos2.lattice,
            self.mos2.lattice,
            None,
            None,
            angles=[0.0],
            nindex=4,
            tol=2e-3,
            lin_tol=2e-3,
            atom_count1=self.mos2.natoms,
            atom_count2=self.mos2.natoms,
            max_atoms=200,
            vector_strain_tol=2e-3,
            dedupe=False,
        )
        reference = lattice.deduplicate_candidates(full_candidates)
        reference.sort(key=lambda item: (item.strain_avg, item.total_atoms, item.angle_deg, item.vector_product))

        optimized = finder.find_supercells(
            self.mos2.lattice,
            self.mos2.lattice,
            None,
            None,
            angles=[0.0],
            nindex=4,
            tol=2e-3,
            lin_tol=2e-3,
            atom_count1=self.mos2.natoms,
            atom_count2=self.mos2.natoms,
            max_atoms=200,
            vector_strain_tol=2e-3,
            dedupe=True,
        )

        self.assertEqual(
            [finder.candidate_to_dict(candidate) for candidate in reference],
            [finder.candidate_to_dict(candidate) for candidate in optimized],
        )

    def test_findn_and_maken_can_generate_an_n_layer_stack(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            nlayer_run = findn.run_findn(
                bottom_poscar=MOS2_PATH,
                bottom_lattice=self.mos2.lattice,
                upper_poscars=[MOS2_PATH, MOS2_PATH],
                upper_lattices=[self.mos2.lattice, self.mos2.lattice],
                bottom_atoms=self.mos2.natoms,
                upper_atoms=[self.mos2.natoms, self.mos2.natoms],
                nindex=12,
                min_angles=[0.0, 0.0],
                max_angles=[60.0, 60.0],
                explicit_angles_by_layer=[[13.15], [13.15]],
                vector_tolerance=2e-3,
                vector_strain_tolerance=2e-3,
                candidate_tolerance=2e-3,
                max_atoms=400,
                output_root=str(temp_root),
            )
            self.assertTrue(nlayer_run.result_path.exists())
            self.assertGreaterEqual(len(nlayer_run.candidates), 1)

            make_run = maken.generate_from_results(
                str(nlayer_run.result_path),
                index=1,
                interlayers=[3.35, 3.35],
                output_dir=str(temp_root),
            )
            self.assertTrue(make_run.output_path.exists())
            self.assertEqual(make_run.total_atoms, 171)

            generated = io.read_poscar(str(make_run.output_path))
            self.assertEqual(generated.natoms, 171)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_supermoire_wrapper_supports_base_independent_and_pairwise_modes(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            tool = cellstine.Supermoire(runs_root=str(temp_root / "runs"), output_root=str(temp_root / "output"))
            independent = tool.findn(
                bottom_poscar=MOS2_PATH,
                upper_poscars=[MOS2_PATH],
                nindex=8,
                match_mode="base_independent",
                min_angles=[0.0],
                max_angles=[60.0],
                explicit_angles_by_layer=[[13.15]],
                vector_tolerance=2e-3,
                vector_strain_tolerance=2e-3,
                candidate_tolerance=2e-3,
                max_atoms=300,
                workers=1,
            )
            self.assertIn("results_dat_upper_1", independent.artifacts)
            self.assertTrue(Path(independent.artifacts["results_dat_upper_1"]).exists())

            pairwise = tool.findn(
                bottom_poscar=MOS2_PATH,
                upper_poscars=[MOS2_PATH],
                nindex=8,
                match_mode="pairwise",
                min_angles=[0.0],
                max_angles=[60.0],
                explicit_angles_by_layer=[[13.15]],
                vector_tolerance=2e-3,
                vector_strain_tolerance=2e-3,
                candidate_tolerance=2e-3,
                max_atoms=300,
                workers=1,
            )
            self.assertEqual(pairwise.summary["pair_count"], 1)
            self.assertIn("results_dat_pair_1", pairwise.artifacts)
            self.assertTrue(Path(pairwise.artifacts["results_dat_pair_1"]).exists())
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_moire_translate_and_translaten_shift_only_the_top_group(self):
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

            moire_tool = MoireWorkflow(runs_root=str(temp_root / "runs"), output_root=str(temp_root / "output"))
            shifted = moire_tool.translate(poscar_path=str(stacked_path), shift_direct=[0.25, 0.0, 0.0])
            shifted_record = io.read_poscar(str(shifted.artifacts["output_poscar"]))
            self.assertTrue(np.allclose(shifted_record.positions_direct[:2], np.array([[0.1, 0.1, 0.10], [0.6, 0.6, 0.18]])))
            self.assertTrue(np.allclose(shifted_record.positions_direct[2:, 0], np.array([0.45, 0.95])))

            super_tool = cellstine.Supermoire(runs_root=str(temp_root / "runs2"), output_root=str(temp_root / "output2"))
            shifted_n = super_tool.translaten(poscar_path=str(stacked_path), shift_direct=[0.0, 0.25, 0.0])
            shifted_n_record = io.read_poscar(str(shifted_n.artifacts["output_poscar"]))
            self.assertTrue(np.allclose(shifted_n_record.positions_direct[:2], np.array([[0.1, 0.1, 0.10], [0.6, 0.6, 0.18]])))
            self.assertTrue(np.allclose(shifted_n_record.positions_direct[2:, 1], np.array([0.45, 0.95])))
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_cli_help_text_mentions_matrix_filter(self):
        parser = moire_cli.build_parser()
        moire_group = parser._subparsers._group_actions[0].choices["moire"]
        help_text = moire_group._subparsers._group_actions[0].choices["find"].format_help()
        self.assertIn("matrix-values", help_text)
        self.assertIn("bilayer commensurate candidates", help_text)
        self.assertIn("workers", help_text)
        self.assertIn("progress", help_text)

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
        self.assertIn("findn", moire_choices)
        self.assertIn("maken", moire_choices)
        self.assertIn("visualize", moire_choices)

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
        with mock.patch("cellstine.cli.interactive.run_interactive", return_value=0) as patched:
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

    def test_candidate_preview_sorts_by_lowest_strain_and_uses_percent_units(self):
        preview = format_bilayer_candidates(
            [
                {
                    "idx": 7,
                    "angle": 12.0,
                    "strain_avg": 0.02,
                    "strain1": 0.02,
                    "strain2": 0.02,
                    "atoms": 80,
                    "ratio1": 1,
                    "ratio2": 1,
                    "i11": 1,
                    "i12": 0,
                    "i21": 0,
                    "i22": 1,
                    "j11": 1,
                    "j12": 0,
                    "j21": 0,
                    "j22": 1,
                },
                {
                    "idx": 3,
                    "angle": 6.0,
                    "strain_avg": 0.001,
                    "strain1": 0.001,
                    "strain2": 0.001,
                    "atoms": 40,
                    "ratio1": 1,
                    "ratio2": 1,
                    "i11": 1,
                    "i12": 0,
                    "i21": 0,
                    "i22": 1,
                    "j11": 1,
                    "j12": 0,
                    "j21": 0,
                    "j22": 1,
                },
            ],
            limit=1,
        )
        self.assertIn("strain_avg(%)", preview)
        self.assertIn("   3", preview)
        self.assertNotIn("   7", preview)

    def test_adsorption_site_preview_shows_direct_and_cartesian_coordinates(self):
        sites = [
            surface.AdsorptionSite("top", (0.25, 0.5, 0.75), (1.0, 2.0, 3.0)),
        ]
        preview = format_adsorption_sites(sites)
        self.assertIn("direct (u, v, w)", preview)
        self.assertIn("cartesian (x, y, z) Ang", preview)
        self.assertIn("top", preview)

    def test_moire_workflow_wrapper_writes_manifest(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            tool = MoireWorkflow(runs_root=str(temp_root / "runs"), output_root=str(temp_root / "output"))
            result = tool.find(
                top_poscar=MOS2_PATH,
                bottom_poscar=MOS2_PATH,
                nindex=4,
                explicit_angles=[13.15],
                max_atoms=200,
                workers=1,
            )
            self.assertTrue(Path(result.manifest_path).exists())
            self.assertIn("results_dat", result.artifacts)
            self.assertIn("timings_s", result.payload)
            self.assertIn("angle_shortlist_s", result.payload["timings_s"])
            self.assertIn("supercell_search_s", result.payload["timings_s"])
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

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
            result = cli_main.execute_namespace(namespace)
            self.assertIn("symmetry_preview", result.payload)
            self.assertEqual(result.summary["backend"], "native")
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
            result = cli_main.execute_namespace(namespace)
            self.assertTrue(Path(result.artifacts["output_poscar"]).exists())
            self.assertEqual(result.summary["cell"], "conventional")
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
            result = cli_main.execute_namespace(namespace)
            self.assertIn("defect_preview", result.payload)
            self.assertIn("atom_001", result.payload["defect_preview"])
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

    def test_moire_prefiltered_matching_equivalence(self):
        from cellstine.moire import lattice as lat
        lat1 = np.array([[2.46, 0.0, 0.0], [-1.23, 2.13, 0.0], [0.0, 0.0, 10.0]])
        lat2 = np.array([[3.15, 0.0, 0.0], [-1.575, 2.728, 0.0], [0.0, 0.0, 10.0]])

        # Test original brute force vs precomputed candidates in find_coincident_vector_pairs
        nindex = 10
        tol = 0.05
        strain_tol = 0.05

        # Brute force
        res_brute = lat.find_coincident_vector_pairs(
            lat1, lat2, nindex, tol, strain_tolerance=strain_tol, precomputed_candidates=None
        )

        # Precomputed candidates
        # 1. Precompute norms and mismatch
        coeffs1, vectors1 = lat.enumerate_in_plane_vectors(lat1, nindex)
        coeffs2, vectors2 = lat.enumerate_in_plane_vectors(lat2, nindex)
        norms1 = np.linalg.norm(vectors1, axis=1)
        norms2 = np.linalg.norm(vectors2, axis=1)
        length_mismatch = np.abs(norms1[:, None] - norms2[None, :]) / np.maximum((norms1[:, None] + norms2[None, :]) * 0.5, 1e-12)
        limit = 2.0 * tol
        candidate_mask = length_mismatch <= limit
        match_rows, match_cols = np.nonzero(candidate_mask)
        precomputed = (match_rows, match_cols, norms1, norms2, length_mismatch)

        res_opt = lat.find_coincident_vector_pairs(
            lat1, lat2, nindex, tol, strain_tolerance=strain_tol, precomputed_candidates=precomputed
        )

        # Assert they yield the exact same matches
        self.assertEqual(len(res_brute), len(res_opt))
        set_brute = {(m.layer1_coeffs, m.layer2_coeffs) for m in res_brute}
        set_opt = {(m.layer1_coeffs, m.layer2_coeffs) for m in res_opt}
        self.assertEqual(set_brute, set_opt)

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

    def test_substrate_stack_uses_sufficient_c_axis_and_requested_gap(self):
        coef = {
            "angle": 43.3139,
            "ratio1": 1,
            "ratio2": 11,
            "i11": -1,
            "i12": 0,
            "i21": 0,
            "i22": -1,
            "j11": -3,
            "j12": -2,
            "j21": 4,
            "j22": -1,
        }

        lattice_out, positions_direct, counts, species, _ = generator.build_supercell(
            RELAXED_OVERLAYER_PATH,
            CU110_PATH,
            coef,
            interlayer_distance=3.35,
            preserve_layer="2",
        )

        expanded_species = []
        for symbol, count in zip(species, counts):
            expanded_species.extend([symbol] * int(count))

        cartesian = io.direct_to_cartesian(positions_direct, lattice_out)
        species_array = np.array(expanded_species)
        top_mask = species_array != "Cu"
        bottom_mask = species_array == "Cu"

        self.assertTrue(np.any(top_mask))
        self.assertTrue(np.any(bottom_mask))

        gap = float(cartesian[top_mask, 2].min() - cartesian[bottom_mask, 2].max())
        self.assertAlmostEqual(gap, 3.35, places=6)

        output_c = float(np.linalg.norm(lattice_out[2]))
        reference_c = max(
            float(np.linalg.norm(self.relaxed_overlayer.lattice[2])),
            float(np.linalg.norm(self.cu110.lattice[2])),
        )
        self.assertGreaterEqual(output_c + 1e-9, reference_c)

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
        coef = {
            "angle": 43.3139,
            "ratio1": 1,
            "ratio2": 11,
            "i11": -1,
            "i12": 0,
            "i21": 0,
            "i22": -1,
            "j11": -3,
            "j12": -2,
            "j21": 4,
            "j22": -1,
        }

        lattice_out, positions_direct, counts, species, flags = generator.build_supercell(
            RELAXED_OVERLAYER_PATH,
            CU110_PATH,
            coef,
            interlayer_distance=3.35,
            preserve_layer="2",
        )

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
            self.assertEqual(selection.molecule_atom_count, self.relaxed_overlayer.natoms)
            self.assertEqual(selection.substrate_atom_count, 11 * self.cu110.natoms)

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
        coef = {
            "angle": 43.3139,
            "ratio1": 1,
            "ratio2": 11,
            "i11": -1,
            "i12": 0,
            "i21": 0,
            "i22": -1,
            "j11": -3,
            "j12": -2,
            "j21": 4,
            "j22": -1,
        }

        lattice_out, positions_direct, counts, species, flags = generator.build_supercell(
            RELAXED_OVERLAYER_PATH,
            CU110_PATH,
            coef,
            interlayer_distance=3.35,
            preserve_layer="2",
        )

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

            manual_cutoff = float(selection.z_cutoff + 10.0)
            manual_selection = molecule.identify_top_group(stacked, z_cutoff=manual_cutoff)
            expected_manual_count = int(np.count_nonzero(z_values > manual_cutoff))

            self.assertEqual(manual_selection.molecule_atom_count, expected_manual_count)
            self.assertNotEqual(manual_selection.molecule_atom_count, selection.molecule_atom_count)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_molecule_reframe_handles_boundary_crossing_adsorbate(self):
        coef = {
            "angle": 43.3139,
            "ratio1": 1,
            "ratio2": 11,
            "i11": -1,
            "i12": 0,
            "i21": 0,
            "i22": -1,
            "j11": -3,
            "j12": -2,
            "j21": 4,
            "j22": -1,
        }

        lattice_out, positions_direct, counts, species, flags = generator.build_supercell(
            RELAXED_OVERLAYER_PATH,
            CU110_PATH,
            coef,
            interlayer_distance=3.35,
            preserve_layer="2",
        )

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

            self.assertEqual(selection.molecule_atom_count, self.relaxed_overlayer.natoms)
            self.assertTrue(np.allclose(selection.center_of_mass_cartesian, run.center_of_mass_after, atol=1e-6))
            self.assertTrue(np.allclose(selection.center_of_mass_cartesian, run.target_cartesian, atol=1e-6))
            self.assertLessEqual(float(np.max(molecule_direct[:, 0]) - np.min(molecule_direct[:, 0])), 1.0 + 1e-8)
            self.assertLessEqual(float(np.max(molecule_direct[:, 1]) - np.min(molecule_direct[:, 1])), 1.0 + 1e-8)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_layer_stage_shifts_only_the_upper_group(self):
        coef = {
            "angle": 43.3139,
            "ratio1": 1,
            "ratio2": 11,
            "i11": -1,
            "i12": 0,
            "i21": 0,
            "i22": -1,
            "j11": -3,
            "j12": -2,
            "j21": 4,
            "j22": -1,
        }

        lattice_out, positions_direct, counts, species, flags = generator.build_supercell(
            RELAXED_OVERLAYER_PATH,
            CU110_PATH,
            coef,
            interlayer_distance=3.35,
            preserve_layer="2",
        )

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

    def test_visualize_writes_html_for_bilayer_and_nlayer_results(self):
        temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            bilayer_run = find.run_find(
                top_poscar=MOS2_PATH,
                bottom_poscar=MOS2_PATH,
                top_lattice=self.mos2.lattice,
                bottom_lattice=self.mos2.lattice,
                top_atoms=self.mos2.natoms,
                bottom_atoms=self.mos2.natoms,
                nindex=12,
                explicit_angles=[13.15],
                vector_tolerance=2e-3,
                vector_strain_tolerance=2e-3,
                candidate_tolerance=2e-3,
                max_atoms=200,
                output_root=str(temp_root),
            )
            bilayer_html = temp_root / "bilayer_viewer.html"
            bilayer_view = visualize.build_visualization(
                str(bilayer_run.dat_path),
                indices=[1],
                output_path=str(bilayer_html),
            )
            self.assertEqual(bilayer_view.results_type, "bilayer")
            self.assertTrue(bilayer_html.exists())
            self.assertIn("CELLSTINE Visualizer", bilayer_html.read_text(encoding="utf-8"))

            nlayer_run = findn.run_findn(
                bottom_poscar=MOS2_PATH,
                bottom_lattice=self.mos2.lattice,
                upper_poscars=[MOS2_PATH, MOS2_PATH],
                upper_lattices=[self.mos2.lattice, self.mos2.lattice],
                bottom_atoms=self.mos2.natoms,
                upper_atoms=[self.mos2.natoms, self.mos2.natoms],
                nindex=12,
                min_angles=[0.0, 0.0],
                max_angles=[60.0, 60.0],
                explicit_angles_by_layer=[[13.15], [13.15]],
                vector_tolerance=2e-3,
                vector_strain_tolerance=2e-3,
                candidate_tolerance=2e-3,
                max_atoms=400,
                output_root=str(temp_root),
            )
            nlayer_html = temp_root / "nlayer_viewer.html"
            nlayer_view = visualize.build_visualization(
                str(nlayer_run.result_path),
                indices=[1],
                output_path=str(nlayer_html),
            )
            self.assertEqual(nlayer_view.results_type, "nlayer")
            self.assertTrue(nlayer_html.exists())
            self.assertIn("3-layer commensurate twist sequence", nlayer_html.read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
