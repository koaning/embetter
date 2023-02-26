import numpy as np
from typing import Union

import spacy
from spacy.language import Language

from embetter.base import EmbetterBase


class spaCyEncoder(EmbetterBase):
    """
    **Usage**

    ```python
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression

    from embetter.grab import ColumnGrabber
    from embetter.text import spaCyEncoder

    # Let's suppose this is the input dataframe
    dataf = pd.DataFrame({
        "text": ["positive sentiment", "super negative"],
        "label_col": ["pos", "neg"]
    })

    # This pipeline grabs the `text` column from a dataframe
    # which is then passed to the medium spaCy model.
    text_emb_pipeline = make_pipeline(
        ColumnGrabber("text"),
        spaCyEncoder("en_core_web_md")
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

    def __init__(self, nlp: Union[str, Language], agg: str = "base"):
        if isinstance(nlp, str):
            self.nlp = spacy.load(nlp, disable=["ner", "tagger", "parser"])
        elif isinstance(nlp, Language):
            self.nlp = nlp
        else:
            raise ValueError("`nlp` must be `str` or spaCy-language object.")
        self.agg = agg

    def fit(self, X, y=None):
        """No-op. Merely checks for object inputs per sklearn standard."""
        # Scikit-learn also expects this in the `.fit()` command.
        self._check_inputs(X)
        return self

    def _check_inputs(self, X):
        options = ["mean", "max", "both", "base"]
        if self.agg not in options:
            raise ValueError(f"The `agg` value must be in {options}. Got {self.agg}.")

    def transform(self, X, y=None):
        """Transforms the phrase text into a numeric representation."""
        self._check_inputs(X)
        docs = self.nlp.pipe(X)
        if self.agg == "base":
            return np.array([d.vector for d in docs])
        token_vectors = [np.array([tok.vector for tok in doc]) for doc in docs]
        if self.agg == "mean":
            return np.array([v.mean(axis=0) for v in token_vectors])
        if self.agg == "max":
            return np.array([v.max(axis=0) for v in token_vectors])
        if self.agg == "both":
            mean_arr = np.array([v.mean(axis=0) for v in token_vectors])
            max_arr = np.array([v.max(axis=0) for v in token_vectors])
            return np.concatenate([mean_arr, max_arr], axis=1)
