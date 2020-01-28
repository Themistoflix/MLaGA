import Code.Nanoparticle as NP
import Code.FCCLattice as FCC
import Code.CuttingPlaneUtilities as CPU
import Code.CutAndSpliceOperator as CPO
import Code.GlobalFeatureClassifier as GFC
import Code.MutationOperator as MO
import Code.PermutationOperator as PO
import Code.EnergyCalculator as EC
import Code.GA.Niched2DPopulation as NIP

from ase.visualize import view
import numpy as np
import matplotlib.pyplot as plt
import copy
import dill
import pickle
import sklearn.gaussian_process as gp
from scipy.spatial import ConvexHull

N_ATOMS = 79
symbols = ['Pt', 'Au']
lattice = FCC.FCCLattice(15, 15, 15, 2)

test_particle = NP.Nanoparticle(lattice)
test_particle.kozlov_sphere(5, symbols, [5, 74])
mixing_energy_calculator = EC.MixingEnergyCalculator()
mixing_energy_calculator.compute_mixing_parameters(test_particle, symbols)

feature_classifier = GFC.SimpleFeatureClassifier(symbols[0], symbols[1])


def initialize_start_population():
    start_population = NIP.Niched2DPopulation(79, 79, 'Pt')
    for n_pt_atoms in range(0, N_ATOMS + 1):
        niche = list()
        for i in range(1):
            p = NP.Nanoparticle(lattice)
            p.kozlov_sphere(5, symbols, [n_pt_atoms, N_ATOMS - n_pt_atoms])
            mixing_energy_calculator.compute_energy(p)
            feature_classifier.compute_feature_vector(p)
            start_population.add_offspring(p)

    return start_population


def locate_convex_hull(start_population, unsuccessful_gens_for_convergence, energy_calculator, energy_key):
    gens_since_last_success = 0

    mutation_operator = MO.MutationOperator(0.5)
    permutation_operator = PO.PermutationOperator(0.5)

    cutting_plane_generator = CPU.SphericalCuttingPlaneGenerator(0.0, 10.0)
    cut_and_splice_operator = CPO.CutAndSpliceOperator(cutting_plane_generator)

    population = copy.deepcopy(start_population)

    energy_log = dict()
    for n_pt_atoms in range(1, N_ATOMS):
        energy_log[n_pt_atoms] = [population[79][n_pt_atoms][0].get_energy(energy_key)]

    # feature_vector_log = dict()
    # for n_pt_atoms in range(1, N_ATOMS):
    # feature_vector_log[n_pt_atoms] = [population[n_pt_atoms][0].get_feature_vector()]

    cur_generation = 0
    while gens_since_last_success < unsuccessful_gens_for_convergence:
        cur_generation += 1
        print("Current generation: {0}".format(cur_generation))
        print(" ")
        print(['-' * 40])
        print(" ")

        if cur_generation % 200 == 0:
            pickle.dump(energy_log, open("energy_log.pkl", 'wb'))

        # choose the new particle from a priority region
        priority_compositions = list(range(N_ATOMS + 1))  # determine_priority_compositions(energy_log, 0.9)
        print("Priority: {0}".format(priority_compositions))

        p = np.random.random()
        if p < 0.6:
            # choose two parents for cut and splice
            print("Cut and Splice")

            while True:
                parents = population.gaussian_tournament(2, 5, (79, np.random.choice(priority_compositions, 1)[0]))
                parent1 = parents[0]
                parent2 = parents[1]
                new_particle = cut_and_splice_operator.cut_and_splice(parent1, parent2)

                if new_particle.get_stoichiometry()['Pt'] in priority_compositions:
                    break

        elif p < 0.8:
            # random permutation
            print("random permutation")
            while True:
                parent = population.random_selection(1)[0]
                new_particle = permutation_operator.random_permutation(parent)

                if new_particle.get_stoichiometry()['Pt'] in priority_compositions:
                    break

        else:
            # random mutation
            print("random mutation")
            while True:
                parent = population.gaussian_tournament(1, 5)[0]
                new_particle = mutation_operator.random_mutation(parent)

                if new_particle.get_stoichiometry()['Pt'] in priority_compositions:
                    break

        # check that it is not a pure particle
        new_stoichiometry = new_particle.get_stoichiometry()
        niche = new_stoichiometry['Pt']

        if niche == N_ATOMS or niche == 0:
            continue

        print("New Stoichiometry: {0}".format(new_stoichiometry))

        # check that it has a unique feature vector
        # feature_classifier.compute_feature_vector(new_particle)
        # unique = True
        # for feature_vector in feature_vector_log[niche]:
        # if np.array_equal(feature_vector, new_particle.get_feature_vector()):
        # unique = False
        # break

        # if not unique:
        # gens_since_last_success += 1
        # continue

        # feature_vector_log[niche].append(new_particle.get_feature_vector())

        # compute energy
        energy_calculator.compute_energy(new_particle)
        print("Energy: {0} = {1}".format(energy_key, new_particle.get_energy(energy_key)))

        # add new offspring to population
        successfull_offspring = False
        peers = population[79][niche]
        peers.sort(key=lambda x: x.get_energy(energy_key), reverse=True)
        if new_particle.get_energy(energy_key) - peers[-1].get_energy(energy_key) < -1e-6:
            print("success: {0}".format(new_particle.get_energy(energy_key) - peers[-1].get_energy(energy_key)))

            successfull_offspring = True
            peers.append(new_particle)

        population[79][niche] = peers

        # reset counters and log energy
        if successfull_offspring:
            gens_since_last_success = 0
            energy_log[niche].append(new_particle.get_energy(energy_key))
        else:
            gens_since_last_success += 1
            energy_log[niche].append(energy_log[niche][-1])

    return population, energy_log


if __name__ == '__main__':
    start_population = initialize_start_population()

    # %%capture
    n_convergence = 100

    n_runs = 1
    populations = list()
    logs = list()

    for run in range(n_runs):
        # start_population = initialize_start_population()
        # gpr_calculator = init_gpr(start_population)
        population, log = locate_convex_hull(start_population, n_convergence, mixing_energy_calculator, 'Mixing Energy')
        populations.append(population)
        logs.append(log)


