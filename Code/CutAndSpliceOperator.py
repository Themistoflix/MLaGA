from Code import Nanoparticle as NP
from Code.IndexedAtoms import IndexedAtoms


class CutAndSpliceOperator:
    def __init__(self, cutting_plane_generator, optimize_coordination_numbers):
        self.cutting_plane_generator = cutting_plane_generator
        self.optimize_coordination_numbers = optimize_coordination_numbers

    def cut_and_splice(self, particle1, particle2):
        self.cutting_plane_generator.set_center(particle1.bounding_box.get_center())
        common_lattice = particle1.lattice

        # make sure that we actually cut
        while True:
            cutting_plane = self.cutting_plane_generator.generate_new_cutting_plane()
            atom_indices_in_positive_subspace, _ = cutting_plane.split_atom_indices(common_lattice, particle1.atoms.get_indices())
            _, atom_indices_in_negative_subspace = cutting_plane.split_atom_indices(common_lattice, particle2.atoms.get_indices())

            if len(atom_indices_in_negative_subspace) > 0 and len(atom_indices_in_positive_subspace) > 0:
                break

        new_atom_data = IndexedAtoms()
        new_atom_data.add_atoms(particle1.get_atoms(atom_indices_in_positive_subspace))
        new_atom_data.add_atoms(particle2.get_atoms(atom_indices_in_negative_subspace))

        new_particle = NP.Nanoparticle(common_lattice)
        new_particle.from_particle_data(new_atom_data)

        if self.optimize_coordination_numbers:
            new_particle.optimize_coordination_numbers()

        # old_stoichiometry = particle1.getStoichiometry()
        # new_particle.enforceStoichiometry(old_stoichiometry)
        return new_particle


