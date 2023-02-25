import numpy as np
from sense2vec import Sense2Vec

from embetter.base import BaseEstimator


class Sense2VecEncoder(BaseEstimator):
    """
    Create a [Sense2Vec encoder](https://github.com/explosion/sense2vec), meant to
    help when encoding phrases as opposed to sentences.

    Arguments:
        path: path to downloaded model

    **Usage**

    ```python
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression

    from embetter.grab import ColumnGrabber
    from embetter.text import Sense2VecEncoder

    # Let's suppose this is the input dataframe
    dataf = pd.DataFrame({
        "text": ["positive sentiment", "super negative"],
        "label_col": ["pos", "neg"]
    })

    # This pipeline grabs the `text` column from a dataframe
    # which is then passed to the sense2vec model.
    text_emb_pipeline = make_pipeline(
        ColumnGrabber("text"),
        Sense2VecEncoder("path/to/s2v")
    )
    X = text_emb_pipeline.fit_transform(dataf, dataf['label_col'])
    ```
    """

    def __init__(self, path: str):
        self.path = path
        self.s2v = Sense2Vec().from_disk(self.path)
        self.shape = self.s2v["duck|NOUN"].shape

    def _to_vector(self, text):
        sense = self.s2v.get_best_sense(text)
        if not sense:
            return np.zeros(shape=self.shape)
        return self.s2v[sense]

    def transform(self, X, y=None):
        """Transforms the phrase text into a numeric representation."""
        return np.array([self._to_vector(x) for x in X])
