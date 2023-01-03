import numpy as np
from embetter.base import EmbetterBase


def _batch(iterable, n=1):
    len_iter = len(iterable)
    for ndx in range(0, len_iter, n):
        yield iterable[ndx : min(ndx + n, len_iter)]


class OpenAIEncoder(EmbetterBase):
    """
    Encoder that can numerically encode sentences.

    Note that this is an **external** embedding provider. If their API breaks, so will this component.
    We also assume that you've already importen openai upfront and ran this command:

    ```python
    import openai

    openai.organization = OPENAI_ORG
    openai.api_key = OPENAI_KEY
    ```

    Arguments:
        model: name of model, can be "small" or "large"
        batch_size: Batch size to send to OpenAI.

    **Usage**:

    ```python
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression

    from embetter.grab import ColumnGrabber
    from embetter.external import CohereEncoder

    import openai

    # You must run this first! 
    openai.organization = OPENAI_ORG
    openai.api_key = OPENAI_KEY

    # Let's suppose this is the input dataframe
    dataf = pd.DataFrame({
        "text": ["positive sentiment", "super negative"],
        "label_col": ["pos", "neg"]
    })

    # This pipeline grabs the `text` column from a dataframe
    # which then get fed into Cohere's endpoint
    text_emb_pipeline = make_pipeline(
        ColumnGrabber("text"),
        OpenAIEncoder()
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

    def __init__(self, model="text-embedding-ada-002", batch_size=25):
        self.model = model
        self.batch_size = batch_size

    def transform(self, X, y=None):
        """Transforms the text into a numeric representation."""
        result = []
        for b in _batch(X, self.batch_size):
            resp = openai.Embedding.create(input=X, model=self.model)
            result.extend([_['embedding'] for _ in resp['data']])
        return np.array(result)
