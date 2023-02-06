import numpy as np
from pathlib import Path

from bpemb import BPEmb

from embetter.base import BaseEstimator


class BytePairEncoder(BaseEstimator):
    """
    This language represents token-free pre-trained subword embeddings. Originally created by
    Benjamin Heinzerling and Michael Strube.

    These vectors will auto-download by the [BPEmb package](https://nlp.h-its.org/bpemb/).
    You can also specify "multi" to download multi language embeddings. A full list of available
    languages can be found [here](https://nlp.h-its.org/bpemb). The article that
    belongs to this work can be found [here](http://www.lrec-conf.org/proceedings/lrec2018/pdf/1049.pdf)
    The availability of vocabulary size as well as dimensionality can be varified
    on the project website. See [here](https://nlp.h-its.org/bpemb/en/) for an
    example link in English. Please credit the original authors if you use their work.

    Arguments:
        lang: name of the model to load
        vs: vocabulary size of the byte pair model
        dim: the embedding dimensionality
        agg: the aggregation method to reduce many subword vectors into a single one, can be "max", "mean" or "both"
        cache_dir: The folder in which downloaded BPEmb files will be cached

    ![](https://raw.githubusercontent.com/koaning/embetter/main/docs/images/sense2vec.png)

    Arguments:
        path: path to downloaded model
    """

    def __init__(self, lang: str, vs: int=10000, dim: int=100, agg:str = "mean", cache_dir: Path = Path.home() / Path(".cache/bpemb")):
        self.lang = lang
        self.vs = vs
        self.dim = dim
        self.cache_dir = cache_dir
        self.agg = agg
        self.module = BPEmb(lang=lang, vs=vs, dim=dim, cache_dir=cache_dir)

    def fit(self, X, y):
        """No-op. Merely checks for object inputs per sklearn standard."""
        # Scikit-learn also expects this in the `.fit()` command.
        self._check_inputs(X)
        return self 

    def _check_inputs(self, X):
        options = ["mean", "max", "both"]
        if self.agg not in options:
            raise ValueError(f"The `agg` value must be in {options}. Got {self.agg}.")
        
    def transform(self, X, y=None):
        """Transforms the phrase text into a numeric representation."""
        self._check_inputs(X)
        if self.agg == "mean":
            return np.array([self.module.embed(x).mean(axis=0) for x in X])
        if self.agg == "max":
            return np.array([self.module.embed(x).max(axis=0) for x in X])
        if self.agg == "both":
            mean_arr = np.array([self.module.embed(x).max(axis=0) for x in X])
            max_arr = np.array([self.module.embed(x).max(axis=0) for x in X])
            return np.concatenate([mean_arr, max_arr], axis=1)
