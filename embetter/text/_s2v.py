import numpy as np
from sense2vec import Sense2Vec

from embetter.base import BaseEstimator


class Sense2VecEncoder(BaseEstimator):
    """
    Create a Sense2Vec encoder. This can be very useful when encoding phrases.

    More information can be found here:
    https://github.com/explosion/sense2vec

    Arguments:
    - path: path to downloaded model
    """

    def __init__(self, path):
        self.s2v = Sense2Vec().from_disk(path)

    def transform(self, X, y=None):
        """Transforms the text (X) into a numeric representation."""
        return np.array([self.s2v[x] for x in X])
