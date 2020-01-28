import numpy as np
from sklearn.cluster import KMeans


class LocalEnvironmentFeatureClassifier:
    def __init__(self):
        pass

    def compute_features_as_index_list(self, particle, recompute_bond_parameters=False):
        if recompute_bond_parameters:
            particle.computeBondParameters()

        n_features = self.compute_n_features(particle)

        features_as_index_lists = list()  # need empty list in case of recalculation
        for i in range(n_features):
            l = list()
            features_as_index_lists.append(l)

        for atomIndex in particle.atoms.get_indices():
            feature = self.predict_atom_feature(particle, atomIndex)
            features_as_index_lists[feature].append(atomIndex)

        particle.setFeaturesAsIndexLists(features_as_index_lists)

    def compute_feature_vector(self, particle, recompute_bond_parameters=False):
        self.compute_features_as_index_list(particle, recompute_bond_parameters)

        n_features = self.compute_n_features(particle)
        feature_vector = np.array([len(particle.getFeaturesAsIndexLists()[feature]) for feature in range(n_features)])

        particle.set_feature_vector(feature_vector)

    def compute_n_features(self, particle):
        raise NotImplementedError

    def predict_atom_feature(self, particle, latticeIndex, recomputeBondParameter=False):
        raise NotImplementedError

    def train(self, training_set):
        raise NotImplementedError


class KMeansClassifier(LocalEnvironmentFeatureClassifier):
    def __init__(self, n_cluster):
        LocalEnvironmentFeatureClassifier.__init__(self)
        self.kMeans = None
        self.n_cluster = n_cluster

    def compute_n_features(self, particle):
        n_elements = len(particle.atoms.get_symbols())
        n_features = self.n_cluster * n_elements
        return n_features

    def predict_atom_feature(self, particle, latticeIndex, recomputeBondParameter=False):
        symbol = particle.atoms.get_symbol(latticeIndex)
        symbols = sorted(particle.atoms.get_symbols())
        symbol_index = symbols.index(symbol)

        offset = symbol_index*self.n_cluster
        if recomputeBondParameter:
            environment = self.kMeans.predict([particle.computeBondParameter(latticeIndex)])[0]
        else:
            environment = self.kMeans.predict([particle.getBondParameter(latticeIndex)])[0]
        return offset + environment

    def train(self, training_set):
        bond_parameters = list()
        for particle in training_set:
            bond_parameters = bond_parameters + particle.getBondParameters()

        print("Starting kMeans")
        self.kMeans = KMeans(n_clusters=self.n_cluster, random_state=0).fit(bond_parameters)



