import numpy as np
import copy


class PermutationOperator:
    def __init__(self, p):
        self.p = p

    def random_permutation(self, particle):
        new_particle = copy.deepcopy(particle)
        if new_particle.is_pure():
            print("Pure particle! No permutation possible")
            return new_particle

        symbols = new_particle.atoms.get_symbols()
        symbol1 = symbols[0]
        symbol2 = symbols[1]

        max_permutations = min(len(new_particle.atoms.get_indices_by_symbol(symbol1)) - 1, len(new_particle.atoms.get_indices_by_symbol(symbol2)) - 1)
        n_permutations = min(self.draw_from_geometric_distribution(), max_permutations)

        symbol1_indices = np.random.choice(new_particle.atoms.get_indices_by_symbol(symbol1), n_permutations, replace=False)
        symbol2_indices = np.random.choice(new_particle.atoms.get_indices_by_symbol(symbol2), n_permutations, replace=False)

        new_particle.atoms.swap_atoms(zip(symbol1_indices, symbol2_indices))

        return new_particle

    def draw_from_geometric_distribution(self):
        return np.random.geometric(p=self.p, size=1)[0]
