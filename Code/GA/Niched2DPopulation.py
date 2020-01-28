from Code.GA.Population import Population
import numpy as np

import matplotlib.pyplot as plt


class Niched2DPopulation(Population):
    def __init__(self, min_n_atoms, max_n_atoms, niche_symbol):
        Population.__init__(self)

        self.min_n_atoms = min_n_atoms
        self.max_n_atoms = max_n_atoms

        self.niche_symbol = niche_symbol
        self.population = dict()
        for n_atoms in range(min_n_atoms, max_n_atoms + 1):
            compositions = dict()
            for niche in range(0, n_atoms + 1):
                compositions[niche] = []

            self.population[n_atoms] = compositions

    def __getitem__(self, item):
        return self.population[item]

    def compute_fitness(self, particle):
        return -1.0*particle.get_energy('EMT')

    def has_allowed_size(self, particle):
        size = particle.get_n_atoms()
        if size in range(self.min_n_atoms, self.max_n_atoms + 1):
            return True
        else:
            return False

    def add_offspring(self, particle):
        size = particle.get_n_atoms()
        niche = particle.get_n_atoms_of_symbol(self.niche_symbol)

        self.population[size][niche].append(particle)

    def random_selection(self, n_individuals):
        selected_offsprings = list()
        possible_slots = [(size, niche) for size in range(self.min_n_atoms, self.max_n_atoms + 1) for niche in range(0, size + 1)]
        selected_slot_indices = np.random.choice(len(possible_slots), n_individuals, replace=False)
        selected_slots = [possible_slots[index] for index in selected_slot_indices]

        for slot in selected_slots:
            size = slot[0]
            niche = slot[1]
            selected_offsprings.append(self.population[size][niche][-1])

        return selected_offsprings

    def tournament_selection(self, n_individuals, tournament_size):
        winners = list()
        for tournament in range(n_individuals):
            candidates = self.random_selection(tournament_size)
            candidates.sort(key=lambda x: x.get_fitness())
            winners.append(candidates[-1])

        return winners

    def gaussian_tournament(self, n_individuals, tournament_size, mean=None):
        winners = list()
        for tournament in range(n_individuals):
            if mean is None:
                size = int(np.random.random() * (self.min_n_atoms - self.max_n_atoms)) + self.min_n_atoms
                niche = int(np.random.random() * size)
                mean = (size, niche)

            sigma = 5
            possible_sizes = range(self.min_n_atoms, self.max_n_atoms + 1)
            probabilities_size = np.array([1/(sigma*2*np.pi)*np.exp(-0.5*((i - mean[0])/sigma)**2) for i in possible_sizes])
            probabilities_size = probabilities_size / np.sum(probabilities_size)

            possible_niches = range(0, self.max_n_atoms + 1)
            probabilities_niche = np.array([1 / (sigma * 2 * np.pi) * np.exp(-0.5 * ((i - mean[1]) / sigma) ** 2) for i in possible_niches])
            probabilities_niche = probabilities_niche / np.sum(probabilities_niche)

            possible_slots = [(size, niche) for size in range(self.min_n_atoms, self.max_n_atoms + 1) for niche in range(0, size + 1)]
            probabilities = []
            for slot in possible_slots:
                size = slot[0]
                niche = slot[1]
                probabilities.append(probabilities_size[size - self.min_n_atoms]*probabilities_niche[niche])
            probabilities = np.array(probabilities)
            probabilities = probabilities/np.sum(probabilities)

            selected_slot_indices = np.random.choice(len(possible_slots), tournament_size, replace=False, p=probabilities)
            selected_slots = [possible_slots[index] for index in selected_slot_indices]

            candidates = list()
            for slot in selected_slots:
                size = slot[0]
                niche = slot[1]
                candidates.append(self.population[size][niche][-1])

            candidates.sort(key=lambda x: x.get_fitness())
            winners.append(candidates[-1])

        return winners

    def get_minima_points(self, energy_key, size=None, niche=None):
        if not size is None:
            points = np.array([np.array([size, niche, self.population[size][niche][-1].get_energy(energy_key)]) for niche in range(0, size + 1)])
        elif not niche is None:
            points = np.array([np.array([size, niche, self.population[size][niche][-1].get_energy(energy_key)]) for size in
                               range(self.min_n_atoms, self.max_n_atoms)])
        elif not niche is None and not size is None:
            points = np.array([np.array[size, niche, self.population[size][niche][-1].get_energy(energy_key)]])
        else:
            points = np.array([np.array([size, niche, self.population[size][niche][-1].get_energy(energy_key)]) for size in range(self.min_n_atoms, self.max_n_atoms) for niche in range(0, size + 1)])

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        return x, y, z

    def plot(self, energy_key, size=None, niche=None):
        x, y, z = self.get_minima_points(energy_key, size, niche)

        if not size is None:
            plt.plot(y, z)
            plt.show()

        elif not niche is None:
            plt.plot(x, z)
            plt.show()
        else:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_trisurf(x, y, z, cmap=plt.cm.viridis, linewidth=0.2)
            plt.show()

