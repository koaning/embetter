from sentence_transformers import SentenceTransformer as SBERT
from embetter.base import BaseEstimator


class SentenceEncoder(BaseEstimator):
    """
    Create a SentenceTransformer.

    You can find the available options here:
    https://www.sbert.net/docs/pretrained_models.html#model-overview

    Arguments:
    - name: name of model, see available options
    """

    def __init__(self, name="all-MiniLM-L6-v2"):
        self.name = name
        self.tfm = SBERT(name)

    def transform(self, X, y=None):
        """Transforms the text (X) into a numeric representation."""
        self.tfm.encode(X)
