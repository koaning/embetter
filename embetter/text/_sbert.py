from sentence_transformers import SentenceTransformer as SBERT
from embetter.base import EmbetterBase


class SentenceEncoder(EmbetterBase):
    """
    Create a SentenceTransformer.

    Arguments:
        name: name of model, see available options
    
    The following model names should be supported: 
    
    - `all-mpnet-base-v2`
    - `multi-qa-mpnet-base-dot-v1`
    - `all-distilroberta-v1`
    - `all-MiniLM-L12-v2`
    - `multi-qa-distilbert-cos-v1`
    - `all-MiniLM-L6-v2`
    - `multi-qa-MiniLM-L6-cos-v1`
    - `paraphrase-multilingual-mpnet-base-v2`
    - `paraphrase-albert-small-v2`
    - `paraphrase-multilingual-MiniLM-L12-v2`
    - `paraphrase-MiniLM-L3-v2`
    - `distiluse-base-multilingual-cased-v1`
    - `distiluse-base-multilingual-cased-v2`

    You can find the more options, and information, on the [sentence-transformers docs page](https://www.sbert.net/docs/pretrained_models.html#model-overview).
    """

    def __init__(self, name="all-MiniLM-L6-v2"):
        self.name = name
        self.tfm = SBERT(name)

    def transform(self, X, y=None):
        """Transforms the text (X) into a numeric representation."""
        return self.tfm.encode(X)
