import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import numpy as np

import moire_cli
from moire import angles, find, finder, generator, io, lattice, make, molecule


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

    def test_cli_help_text_mentions_matrix_filter(self):
        parser = moire_cli.build_parser()
        help_text = parser._subparsers._group_actions[0].choices["find"].format_help()
        self.assertIn("matrix-values", help_text)
        self.assertIn("commensurate superlattice candidates", help_text)

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


if __name__ == '__main__':
    unittest.main()
