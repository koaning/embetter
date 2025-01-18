from itertools import islice

import numpy as np
from openai import AzureOpenAI, OpenAI

from embetter.base import EmbetterBase


def _batch(iterable, n=1):
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


class OpenAIEncoder(EmbetterBase):
    """
    Encoder that can numerically encode sentences.

    Note that this is an **external** embedding provider. If their API breaks, so will this component.
    We also assume that you've already importen openai upfront and ran this command:

    This encoder will require the `OPENAI_API_KEY` (optionally `OPENAI_ORG_ID` and `OPENAI_PROJECT_ID`) environment variable to be set.
    If you have it defined in your `.env` file, you can use python-dotenv to load it.

    You also need to install the `openai` library beforehand.

    ```
    python -m pip install openai
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
    from embetter.external import OpenAIEncoder
    from dotenv import load_dotenv

    load_dotenv()  # take environment variables from .env.

    # Let's suppose this is the input dataframe
    dataf = pd.DataFrame({
        "text": ["positive sentiment", "super negative"],
        "label_col": ["pos", "neg"]
    })

    # This pipeline grabs the `text` column from a dataframe
    # which then get fed into OpenAI's endpoint
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
        # You must run this first!
        self.client = OpenAI()
        self.model = model
        self.batch_size = batch_size

    def transform(self, X, y=None):
        """Transforms the text into a numeric representation."""
        result = []
        for b in _batch(X, self.batch_size):
            resp = self.client.embeddings.create(input=b, model=self.model)  # fmt: off
            result.extend([_.embedding for _ in resp.data])
        return np.array(result)


class AzureOpenAIEncoder(OpenAIEncoder):
    """
    Encoder that can numerically encode sentences.

    Note that this is an *external* embedding provider. If their API breaks, so will this component.

    To use this encoder you must provide credentials. Please provide one of the `api_key`, `azure_ad_token`, `azure_ad_token_provider` arguments, or the `AZURE_OPENAI_API_KEY` or `AZURE_OPENAI_AD_TOKEN`.
    You must provide one of the `base_url` or `azure_endpoint` arguments, or the `AZURE_OPENAI_ENDPOINT` environment variable.
    Furthermore you must provide either the `api_version` argument or the `OPENAI_API_VERSION` environment variable.

    If you have your enviroment variables defined in your `.env` file, you can use python-dotenv to load it.

    You also need to install the `openai` library beforehand.

    ```
    python -m pip install openai
    ```

    Arguments:
        model: name of model.
        batch_size: Batch size to send to AzureOpenAI.

    *Usage*:

    ```python
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression

    from embetter.grab import ColumnGrabber
    from embetter.external import AzureOpenAIEncoder
    from dotenv import load_dotenv

    load_dotenv()  # take environment variables from .env.

    # Let's suppose this is the input dataframe
    dataf = pd.DataFrame({
        "text": ["positive sentiment", "super negative"],
        "label_col": ["pos", "neg"]
    })

    # This pipeline grabs the `text` column from a dataframe
    # which then get fed into OpenAI's endpoint
    text_emb_pipeline = make_pipeline(
        ColumnGrabber("text"),
        AzureOpenAIEncoder()
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

    def _init_(self, model="text-embedding-ada-002", batch_size=25, **kwargs):
        self.model = model
        self.batch_size = batch_size
        self.client = AzureOpenAI(**kwargs)
