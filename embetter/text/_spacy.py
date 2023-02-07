import numpy as np
from pathlib import Path
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
    # which then get fed into Sentence-Transformers' all-MiniLM-L6-v2.
    text_emb_pipeline = make_pipeline(
        ColumnGrabber("text"),
        spaCyEncoder(lang="en")
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
    def __init__(self, nlp: Union[str, Language], agg:str = "base"):
        if isinstance(nlp, str):
            self.nlp = spacy.load(nlp, deactivate=["ner", "tagger", "parser"])
        elif isinstance(nlp, Language):
            self.nlp = nlp
        else:
            raise ValueError(
                "`nlp` must be `str` or spaCy-language object."
            )
        self.agg = agg

    def transform(self, X, y=None):
        """Transforms the phrase text into a numeric representation."""
        return np.array([d.vector for d in self.nlp.pipe(X)])
        if self.agg == "mean":
            return np.array([self.module.embed(x).mean(axis=0) for x in X])
        if self.agg == "max":
            return np.array([self.module.embed(x).max(axis=0) for x in X])
        if self.agg == "both":
            mean_arr = np.array([self.module.embed(x).max(axis=0) for x in X])
            max_arr = np.array([self.module.embed(x).max(axis=0) for x in X])
            return np.concatenate([mean_arr, max_arr], axis=1)
