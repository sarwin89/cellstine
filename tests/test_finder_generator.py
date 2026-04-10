import shutil
import sys
import unittest
from pathlib import Path
from uuid import uuid4

import numpy as np

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src")))

import cellstine
import moire_cli
from cellstine.interface.surface import Surface as InterfaceSurface
from cellstine.interface.interface import Interface as InterfaceWorkflow
from cellstine.moire.moire import Moire as MoireWorkflow
from moire import angles, find, finder, findn, generator, io, lattice, make, maken, molecule, surface, visualize


BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_DIR = BASE_DIR / 'input'


def _sample_path(filename: str) -> str:
    input_path = INPUT_DIR / filename
    if input_path.exists():
        return str(input_path)
    legacy_path = BASE_DIR / filename
    return str(legacy_path)


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

    def test_cli_help_text_mentions_matrix_filter(self):
        parser = moire_cli.build_parser()
        moire_group = parser._subparsers._group_actions[0].choices["moire"]
        help_text = moire_group._subparsers._group_actions[0].choices["find"].format_help()
        self.assertIn("matrix-values", help_text)
        self.assertIn("bilayer commensurate candidates", help_text)
        self.assertIn("workers", help_text)

    def test_cli_help_text_mentions_grouped_workflows_and_subcommands(self):
        parser = moire_cli.build_parser()
        choices = parser._subparsers._group_actions[0].choices
        self.assertIn("moire", choices)
        self.assertIn("adsorbate", choices)
        self.assertIn("interface", choices)

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

    def test_cellstine_package_exports_public_classes(self):
        self.assertEqual(cellstine.__version__, "4.0.0")
        self.assertTrue(hasattr(cellstine, "Moire"))
        self.assertTrue(hasattr(cellstine, "Molecule"))
        self.assertTrue(hasattr(cellstine, "Interface"))

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
            self.assertEqual(run.total_atoms, 6)

            slab = io.read_poscar(str(run.output_path))
            self.assertEqual(slab.natoms, 6)
            self.assertGreater(float(np.linalg.norm(slab.lattice[2])), 3.0)
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
            self.assertEqual(run.site_counts.get("top"), 8)
            self.assertEqual(run.site_counts.get("bridge"), 24)
            self.assertEqual(run.site_counts.get("hcp_hollow"), 8)
            self.assertEqual(run.site_counts.get("fcc_hollow"), 8)

            slab = io.read_poscar(str(run.output_path))
            report = surface.find_adsorption_sites(slab)
            self.assertAlmostEqual(report.average_top_layer_coordination, 6.0, places=6)
            self.assertEqual(report.site_counts.get("top"), 8)
            self.assertEqual(report.site_counts.get("bridge"), 24)
            self.assertEqual(report.site_counts.get("hcp_hollow"), 8)
            self.assertEqual(report.site_counts.get("fcc_hollow"), 8)
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
