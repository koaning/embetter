
from model2vec import StaticModel

from embetter.base import EmbetterBase


class TextEncoder(EmbetterBase):
    """
    Encoder that can numerically encode text using a model from the model2vec library.

    The main benefit of this encoder is that it uses distilled word embeddings, which means that they are super *fast*. 

    Arguments:
        name: name of model, see available options, can also pass a model2vec StaticModel object directly

    The following model names should be supported:

    - `potion-base-32M`
    - `potion-base-8M`
    - `potion-base-4M`
    - `potion-base-2M`
    - `potion-retrieval-32M`
    - `M2V_multilingual_output`

    You can find the more options, and information, on the [Github repository](https://github.com/MinishLab/model2vec?tab=readme-ov-file#model-list).

    **Usage**:

    ```python
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression

    from embetter.grab import ColumnGrabber
    from embetter.text import TextEncoder

    # Let's suppose this is the input dataframe
    dataf = pd.DataFrame({
        "text": ["positive sentiment", "super negative"],
        "label_col": ["pos", "neg"]
    })

    # This pipeline grabs the `text` column from a dataframe
    # which then get fed into Sentence-Transformers' all-MiniLM-L6-v2.
    text_emb_pipeline = make_pipeline(
        ColumnGrabber("text"),
        TextEncoder()
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
        self, model="potion-base-8M", device=None, quantize=False, num_threads=None
    ):
        if isinstance(model, str):
            self.model = StaticModel.from_pretrained(model)
        else:
            assert isinstance(model, StaticModel), "model must be a string or a StaticModel from model2vec"
            self.model = model

    def transform(self, X, y=None):
        """Transforms the text into a numeric representation."""
        return self.model.encode(X)


