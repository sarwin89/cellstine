import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import numpy as np

from moire import angles, find, finder, io, lattice, make


BASE_DIR = Path(__file__).resolve().parents[1]
MOS2_PATH = str(BASE_DIR / 'mos2.vasp')
GRAPHENE_PATH = str(BASE_DIR / 'graph.vasp')
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
            output_root=str(BASE_DIR / f"moire_test_{uuid4().hex}"),
        )
        try:
            self.assertEqual(find_run.symmetry_lcm, 60)
            self.assertAlmostEqual(find_run.search_min_angle, 0.0, places=8)
            self.assertAlmostEqual(find_run.search_max_angle, 60.0, places=8)
            self.assertIn(0.0, find_run.angle_values)
            self.assertIn(60.0, find_run.angle_values)
        finally:
            shutil.rmtree(find_run.run_dir.parent, ignore_errors=True)

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
            self.assertTrue(find_run.json_path.exists())
            self.assertTrue(find_run.markdown_path.exists())
            self.assertTrue(find_run.dat_path.exists())
            self.assertGreater(len(find_run.candidates), 0)

            make_run = make.generate_from_results(
                str(find_run.json_path),
                index=1,
                interlayer_distance=3.35,
            )
            self.assertTrue(make_run.output_path.exists())
            self.assertGreater(make_run.total_atoms, 0)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_make_matches_reference_counts(self):
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
                payload = {
                    "meta": {
                        "top_poscar": MOS2_PATH,
                        "bottom_poscar": MOS2_PATH,
                    },
                    "candidates": [
                        {
                            "index": 1,
                            "angle_deg": best.angle_deg,
                            "ratio1": best.ratio1,
                            "ratio2": best.ratio2,
                            "layer1_vector1": list(best.layer1_vector1),
                            "layer1_vector2": list(best.layer1_vector2),
                            "layer2_vector1": list(best.layer2_vector1),
                            "layer2_vector2": list(best.layer2_vector2),
                            "strain_avg": best.strain_avg,
                            "strain_layer1": best.strain_layer1,
                            "strain_layer2": best.strain_layer2,
                            "eps1": best.eps1,
                            "eps2": best.eps2,
                            "vector_product": best.vector_product,
                            "area1": best.area1,
                            "area2": best.area2,
                            "total_atoms": best.total_atoms,
                        }
                    ],
                }
                temp_root = BASE_DIR / f"moire_test_{uuid4().hex}"
                temp_root.mkdir(parents=True, exist_ok=False)
                try:
                    json_path = temp_root / "find_results.json"
                    json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
                    run = make.generate_from_results(str(json_path), index=1, interlayer_distance=0.0)
                    generated = io.read_poscar(str(run.output_path))
                    reference = io.read_poscar(reference_path)
                    self.assertEqual(generated.counts, reference.counts)
                finally:
                    shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
