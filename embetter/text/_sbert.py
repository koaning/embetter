from warnings import warn

import pandas as pd
import torch
from torch.nn import Linear
from torch.quantization import quantize_dynamic
from sentence_transformers import SentenceTransformer as SBERT

from embetter.base import EmbetterBase


class SentenceEncoder(EmbetterBase):
    """
    Encoder that can numerically encode sentences.

    Arguments:
        name: name of model, see available options
        device: manually override cpu/mps/gpu device, tries to grab gpu or mps automatically when available
        quantize: turns on quantization
        num_threads: number of treads for pytorch to use, only affects when device=cpu

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

    **Usage**:

    ```python
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression

    from embetter.grab import ColumnGrabber
    from embetter.text import SentenceEncoder

    # Let's suppose this is the input dataframe
    dataf = pd.DataFrame({
        "text": ["positive sentiment", "super negative"],
        "label_col": ["pos", "neg"]
    })

    # This pipeline grabs the `text` column from a dataframe
    # which then get fed into Sentence-Transformers' all-MiniLM-L6-v2.
    text_emb_pipeline = make_pipeline(
        ColumnGrabber("text"),
        SentenceEncoder('all-MiniLM-L6-v2')
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
        self, name="all-MiniLM-L6-v2", device=None, quantize=False, num_threads=None
    ):
        if not device:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.name = name
        self.device = device
        self.tfm = SBERT(name, device=self.device)
        self.num_threads = num_threads
        self.quantize = quantize
        if quantize:
            self.tfm = quantize_dynamic(self.tfm, {Linear})
        if num_threads:
            if self.device.type == "cpu":
                torch.set_num_threads(num_threads)

    def transform(self, X, y=None):
        """Transforms the text into a numeric representation."""
        # Convert pd.Series objects to encode compatable
        if isinstance(X, pd.Series):
            X = X.to_numpy()

        return self.tfm.encode(X)


def MatrouskaEncoder(name="tomaarsen/mpnet-base-nli-matryoshka", **kwargs):
    warn(
        "Please use `MatryoshkaEncoder` instead of `MatrouskaEncoder."
        "We will use correct spelling going forward and `MatrouskaEncoder` will be deprecated.",
        DeprecationWarning,
    )
    return MatryoshkaEncoder(name="tomaarsen/mpnet-base-nli-matryoshka", **kwargs)


def MatryoshkaEncoder(name="tomaarsen/mpnet-base-nli-matryoshka", **kwargs):
    """
    Encoder that can numerically encode sentences.

    This function, which looks like a class, offers a shorthand way to fetch pretrained
    [Matryoshka embeddings](https://www.sbert.net/examples/training/matryoshka/README.html).
    Under the hood it just returns a `SentenceEncoder` object, but the default name points
    to a pretrained Matryoshka model.

    These embeddings are more flexible in the sense that you can more easily reduce the
    dimensions without losing as much information. The aforementioned docs give more details

    **Usage**:

    ```python
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression

    from embetter.grab import ColumnGrabber
    from embetter.text import SentenceEncoder

    # Let's suppose this is the input dataframe
    dataf = pd.DataFrame({
        "text": ["positive sentiment", "super negative"],
        "label_col": ["pos", "neg"]
    })

    # This pipeline grabs the `text` column from a dataframe
    # which then get fed into Sentence-Transformers' all-MiniLM-L6-v2.
    text_emb_pipeline = make_pipeline(
        ColumnGrabber("text"),
        MatryoshkaEncoder()
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
    return SentenceEncoder(name=name, **kwargs)
