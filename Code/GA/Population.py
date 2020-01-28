class Population:
    def __init__(self):
        pass

    def add_offspring(self, particle):
        raise NotImplementedError

    def compute_fitness(self, particle):
        raise NotImplementedError

    def random_selection(self, n_individuals):
        raise NotImplementedError

    def tournament_selection(self, n_individuals, tournament_size):
        raise NotImplementedError

    def gaussian_tournament(self, n_individuals, tournament_size, mean=None):
        raise NotImplementedError
