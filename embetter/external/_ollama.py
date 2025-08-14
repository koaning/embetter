from itertools import islice

import numpy as np
from ollama import Client

from embetter.base import EmbetterBase


def _batch(iterable, n=1):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


class OllamaEncoder(EmbetterBase):
    """
    Encoder that can numerically encode sentences.

    Note that this is an **external** embedding provider. If their API breaks, so will this component.
    We also assume that you've already importen ollama upfront and ran this command:

    You need to install the `ollama` library beforehand.

    ```
    python -m pip install ollama
    ```

    Arguments:
        model: name of model
        batch_size: Batch size to send to Ollama.

    **Usage**:

    ```python
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression

    from embetter.grab import ColumnGrabber
    from embetter.external import OllamaEncoder

    # Let's suppose this is the input dataframe
    dataf = pd.DataFrame({
        "text": ["positive sentiment", "super negative"],
        "label_col": ["pos", "neg"]
    })

    # This pipeline grabs the `text` column from a dataframe
    # which then get fed into Ollama's endpoint
    text_emb_pipeline = make_pipeline(
        ColumnGrabber("text"), OllamaEncoder(model="nomic-embed-text:latest")
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

    def __init__(self, model, host="http://localhost:11434", batch_size=25):
        self.client = Client(host=host)
        self.model = model
        self.batch_size = batch_size

    def transform(self, X, y=None):
        """Transforms the text into a numeric representation."""
        n_rows = len(X)
        n_dims = None
        result = None
        idx = 0

        for b in _batch(X, self.batch_size):
            resp = self.client.embed(model=self.model, input=b)
            batch_embeddings = np.array(resp.embeddings)

            if result is None:
                n_dims = batch_embeddings.shape[1]
                result = np.zeros((n_rows, n_dims), dtype=batch_embeddings.dtype)

            result[idx : idx + len(b)] = batch_embeddings
            idx += len(b)

        return result
