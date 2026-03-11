import os
import unittest

import numpy as np

from moire import angles, finder, generator, io


PYTHON_WORKDIR = os.getcwd()
MOS2_PATH = os.path.join(PYTHON_WORKDIR, 'mos2.vasp')
REFERENCE_FILES = {
    13.15: os.path.join(PYTHON_WORKDIR, 'Results', 'spc_POSCAR_13-B.vasp'),
    21.787: os.path.join(PYTHON_WORKDIR, 'Results', 'spc_POSCAR_21-B.vasp'),
    27.9: os.path.join(PYTHON_WORKDIR, 'Results', 'spc_POSCAR_27-B.vasp'),
}


class FinderGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mos2 = io.read_poscar(MOS2_PATH)

    def test_cellfind_shortlists_reference_angle_families(self):
        candidates = angles.find_commensurate_angles(
            self.mos2.lattice,
            self.mos2.lattice,
            12,
            strain_tolerance=2e-3,
            min_angle=0.0,
            max_angle=30.0,
        )
        found_angles = np.array([candidate.angle_deg for candidate in candidates])
        self.assertTrue(np.any(np.isclose(found_angles, 13.1735, atol=0.02)))
        self.assertTrue(np.any(np.isclose(found_angles, 21.7868, atol=0.02)))
        self.assertTrue(np.any(np.isclose(found_angles, 27.7958, atol=0.02)))

    def test_zero_angle_self_match(self):
        results = finder.find_supercells(
            self.mos2.lattice,
            self.mos2.lattice,
            0.0,
            0.0,
            angle_step=0.1,
            nindex=3,
            tol=1e-6,
            lin_tol=1e-6,
            atom_count1=self.mos2.natoms,
            atom_count2=self.mos2.natoms,
        )
        self.assertTrue(results)
        best = results[0]
        self.assertAlmostEqual(best.angle_deg, 0.0, places=8)
        self.assertAlmostEqual(best.strain_avg, 0.0, places=8)
        self.assertEqual(best.total_atoms, 6)
        self.assertEqual((best.ratio1, best.ratio2), (1, 1))

    def test_reference_angles_atom_counts(self):
        expected_atoms = {
            13.15: 114,
            21.787: 42,
            27.9: 78,
        }
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

    def test_generator_matches_reference_supercells(self):
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
                record = {
                    'i11': best.layer1_vector1[0],
                    'i12': best.layer1_vector1[1],
                    'i21': best.layer1_vector2[0],
                    'i22': best.layer1_vector2[1],
                    'j11': best.layer2_vector1[0],
                    'j12': best.layer2_vector1[1],
                    'j21': best.layer2_vector2[0],
                    'j22': best.layer2_vector2[1],
                    'ratio1': best.ratio1,
                    'ratio2': best.ratio2,
                    'angle': best.angle_deg,
                }
                lattice_new, positions_new, counts_new, species_new, _ = generator.build_supercell(
                    MOS2_PATH,
                    MOS2_PATH,
                    record,
                )
                reference = io.read_poscar(reference_path)
                self.assertEqual(sum(counts_new), reference.natoms)
                self.assertEqual(counts_new, reference.counts)
                self.assertEqual(species_new, reference.species)
                self.assertEqual(positions_new.shape[0], reference.natoms)
                for index in range(2):
                    self.assertAlmostEqual(
                        float(np.linalg.norm(lattice_new[index])),
                        float(np.linalg.norm(reference.lattice[index])),
                        places=5,
                    )


if __name__ == '__main__':
    unittest.main()
