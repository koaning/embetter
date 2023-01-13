import torch
from sentence_transformers import SentenceTransformer as SBERT
from embetter.base import EmbetterBase
import pandas as pd


class SentenceEncoder(EmbetterBase):
    """
    Encoder that can numerically encode sentences.

    ![](https://raw.githubusercontent.com/koaning/embetter/main/docs/images/sentence-encoder.png)

    Arguments:
        name: name of model, see available options
        device: manually override cpu/gpu device, tries to grab gpu automatically when available

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

    def __init__(self, name="all-MiniLM-L6-v2", device=None):
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        self.device = device
        self.tfm = SBERT(name, device=self.device)

    def transform(self, X, y=None):
        """Transforms the text into a numeric representation."""
        # Convert pd.Series objects to encode compatable
        if isinstance(X, pd.Series):
            X = X.to_numpy()

        return self.tfm.encode(X)
