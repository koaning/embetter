import numpy as np
from sense2vec import Sense2Vec

from embetter.base import BaseEstimator


class Sense2VecEncoder(BaseEstimator):
    """
    Create a [Sense2Vec encoder](https://github.com/explosion/sense2vec), meant to
    help when encoding phrases as opposed to sentences.

    ![](https://raw.githubusercontent.com/koaning/embetter/main/docs/images/sense2vec.png)

    Arguments:
        path: path to downloaded model
    """

    def __init__(self, path: str):
        self.path = path
        self.s2v = Sense2Vec().from_disk(self.path)

    def transform(self, X, y=None):
        """Transforms the phrase text into a numeric representation."""
        return np.array([self.s2v[x] for x in X])
