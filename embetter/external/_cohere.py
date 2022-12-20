import numpy as np 
from embetter.base import EmbetterBase

def _batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


class CohereEncoder(EmbetterBase):
    """
    Encoder that can numerically encode sentences.

    Note that this is an **external** embedding provider. If their API breaks, so will this component.

    Arguments:
        client: cohere client with key
        model: name of model, can be "small" or "large"

    **Usage**:

    ```python
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    
    from cohere import Client
    from embetter.grab import ColumnGrabber
    from embetter.external import CohereEncoder

    client = Client("APIKEY")
    # Let's suppose this is the input dataframe
    dataf = pd.DataFrame({
        "text": ["positive sentiment", "super negative"],
        "label_col": ["pos", "neg"]
    })

    # This pipeline grabs the `text` column from a dataframe
    # which then get fed into Cohere's endpoint
    text_emb_pipeline = make_pipeline(
        ColumnGrabber("text"),
        CohereEncoder(client=client, model="large")
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

    def __init__(self, client, model="large"):
        self.client = client
        self.model = model

    def transform(self, X, y=None):
        """Transforms the text into a numeric representation."""
        result = []
        for b in _batch(X, 10):
            response = self.client.embed(b)
            result.extend(response.embeddings)
        return np.array(result)
