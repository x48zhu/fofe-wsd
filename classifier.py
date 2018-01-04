import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier


class Classifier(object):
    def __init__(self):
        self.model = None

    def fit(self, features, targets):
        raise NotImplementedError

    def predict(self, features):
        """
        Template for prediction
        :param features: 2D numpy array
        :return: 1D numpy array
        """
        raise NotImplementedError


class KNNClassifier(Classifier):
    def __init__(self, k, cos_similarity=True):
	super(KNNClassifier, self).__init__()
        self.k = k
        self.cos_similarity = cos_similarity

    def fit(self, features, targets):
        if self.cos_similarity:
            # Features are normalized to unit 1, then cosine simliarity between unit feature
            # is equivalent to their euclidean distance.
            features = features / np.sqrt(np.sum(features*features, axis=1)).reshape(-1,1)
        super(KNNClassifier, self).__init__()
        self.model = KNeighborsClassifier(self.k)
        self.model.fit(features, targets)

    def predict(self, features):
        if self.cos_similarity:
            features = features / np.sqrt(np.sum(features*features, axis=1)).reshape(-1,1)
        return self.model.predict(features)


class AverageClassifier(Classifier):
    def __init__(self, cos_similarity=True):
        super(AverageClassifier, self).__init__()
	self.cos_similarity = cos_similarity

    def fit(self, features, targets):
        avg_features = []
        avg_targets = []
        for target in np.unique(targets):
            local_features = features[np.where(targets==target)]
            avg_features.append(np.mean(local_features, axis=0))
            avg_targets.append(target)
	avg_features = np.array(avg_features)
	avg_targets = np.array(avg_targets)
	if self.cos_similarity:
            # Features are normalized to unit 1, then cosine simliarity between unit feature
            # is equivalent to their euclidean distance.
            avg_features = avg_features / np.sqrt(np.sum(avg_features*avg_features, axis=1)).reshape(-1,1)
        self.model = KNeighborsClassifier(1)
        self.model.fit(avg_features, avg_targets)

    def predict(self, features):
        return self.model.predict(features)


class MostFreqClassifier(Classifier):
    def __init__(self):
        super(MostFreqClassifier, self).__init__()

    def fit(self, features, targets):
        counts = np.bincount(targets)
        self.model = np.argmax(counts)

    def predict(self, features):
        return np.array(self.model * len(features)).reshape(1, -1)


class NNClassifier(Classifier):
    def __init__(self, feature_dim=512):
        super(NNClassifier, self).__init__()
        np.random.seed(7)
        self.model = Sequential()
        self.model.add(Dense(256, input_dim=feature_dim, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, features, targets):
        self.model.fit(features, targets, nb_epoch=50, batch_size=10)

    def predict(self, features):
        return self.model.predict(features)
