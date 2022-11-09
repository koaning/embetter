import numpy as np
from tqdm.auto import tqdm
from sense2vec import Sense2Vec
from embetter.base import BaseEstimator


class Sense2VecEncoder(BaseEstimator):
    """
    Create a [Sense2Vec encoder](https://github.com/explosion/sense2vec), meant to
    help when encoding phrases as opposed to sentences.

    ![](https://raw.githubusercontent.com/koaning/embetter/main/docs/images/sense2vec.png)

    Arguments:
        path: path to downloaded model
        show_progress_bar: Show a progress bar when encoding phrases
    """

    def __init__(self, path, show_progress_bar=False):
        self.s2v = Sense2Vec().from_disk(path)
        self.show_progress_bar = show_progress_bar

    def transform(self, X, y=None):
        """Transforms the phrase text into a numeric representation."""
        instances = tqdm(X, desc="Sense2Vec Encoding") if self.show_progress_bar else X
        output = []
        for x in instances:
            output.append(self.s2v[x])
        return np.array(output)
