from typing import Literal, Union

import numpy as np
from gensim import downloader
from gensim.models import KeyedVectors, Word2Vec
from gensim.utils import tokenize

from embetter.base import EmbetterBase


class Word2VecEncoder(EmbetterBase):
    """
    Encodes text using a static word embedding model. The component uses gensim's default tokenizer.

    Arguments:
        model: Name of model to load from gensim's repositories or a word embedding model, or its keyed vectors.
        agg: Way to aggregate the word embeddings in a document. Can either take the maximum, mean or both of them concatenated.
        deacc: Specifies whether accents should be removed when tokenizing the text.
        lowercase: Specifies whether the text should be lowercased during tokenization.

    Currently the following models are supported by default:
     - `conceptnet-numberbatch-17-06-300`
     - `word2vec-ruscorpora-300`
     - `word2vec-google-news-300`
     - `glove-wiki-gigaword-50`
     - `glove-wiki-gigaword-100`
     - `glove-wiki-gigaword-200`
     - `glove-wiki-gigaword-300`
     - `glove-twitter-25`
     - `glove-twitter-50`
     - `glove-twitter-100`
     - `glove-twitter-200`

    **Usage**

    ```python
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression

    from embetter.grab import ColumnGrabber
    from embetter.text import Word2VecEncoder

    # Let's suppose this is the input dataframe
    dataf = pd.DataFrame({
        "text": ["positive sentiment", "super negative"],
        "label_col": ["pos", "neg"]
    })

    # This pipeline grabs the `text` column from a dataframe
    # which is then passed to a Word2Vec model.
    text_emb_pipeline = make_pipeline(
        ColumnGrabber("text"),
        Word2VecEncoder("glove-wiki-gigaword-50")
    )
    X = text_emb_pipeline.fit_transform(dataf, dataf['label_col'])

    # This pipeline can also be trained to make predictions, using
    # the embedded features.
    text_clf_pipeline = make_pipeline(
        text_emb_pipeline,
        LogisticRegression()
    )

    # Prediction example
    text_clf_pipeline.fit(dataf, dataf['label_col']).predict(dataf)
    ```
    """

    def __init__(
        self,
        model: Union[str, Word2Vec, KeyedVectors] = "word2vec-google-news-300",
        agg: Literal["mean", "max", "both"] = "mean",
        deacc: bool = False,
        lowercase: bool = False,
    ):
        self.model = model
        if isinstance(model, str):
            if model not in downloader.info()["models"]:
                raise ValueError("Model with the specified name not found.")
            self.keyed_vectors: KeyedVectors = downloader.load(model)  # type: ignore
        elif isinstance(model, Word2Vec):
            self.keyed_vectors: KeyedVectors = model.wv
        elif isinstance(model, KeyedVectors):
            self.keyed_vectors: KeyedVectors = model
        else:
            raise TypeError(
                f"You should pass a model name, keyed vectors or a Word2Vec model to Word2VecEncoder, not {type(model)}"
            )
        self.agg = agg
        self.deacc = deacc
        self.lowercase = lowercase

    def fit(self, X, y=None):
        """No-op. Merely checks for object inputs per sklearn standard."""
        # Scikit-learn also expects this in the `.fit()` command.
        self._check_inputs(X)
        self.n_features_out = (
            self.keyed_vectors.vector_size
            if self.agg != "both"
            else self.keyed_vectors.vector_size * 2
        )
        return self

    def _check_inputs(self, X):
        options = ["mean", "max", "both"]
        if self.agg not in options:
            raise ValueError(f"The `agg` value must be in {options}. Got {self.agg}.")

    def _tokenize(self, X) -> list[list[int]]:
        token_indices = []
        for text in X:
            tokens = tokenize(text, deacc=self.deacc, lowercase=self.lowercase)
            indices = []
            for token in tokens:
                index = self.keyed_vectors.get_index(token, default=-1)
                if index != -1:
                    indices.append(index)
            token_indices.append(indices)
        return token_indices

    def transform(self, X, y=None):
        """Transforms the phrase text into a numeric representation using word embeddings."""
        self._check_inputs(X)
        tokens = self._tokenize(X)
        embeddings = np.empty((len(X), self.n_features_out))
        for i_doc, token_indices in enumerate(tokens):
            if not len(token_indices):
                embeddings[i_doc, :] = np.nan
            doc_vectors = self.keyed_vectors.vectors[token_indices]
            if self.agg == "mean":
                embeddings[i_doc, :] = np.mean(doc_vectors, axis=0)
            elif self.agg == "max":
                embeddings[i_doc, :] = np.max(doc_vectors, axis=0)
            elif self.agg == "both":
                print("both")
                mean_vector = np.mean(doc_vectors, axis=0)
                max_vector = np.max(doc_vectors, axis=0)
                embeddings[i_doc, :] = np.concatenate((mean_vector, max_vector))
        return embeddings
