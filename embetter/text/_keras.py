import numpy as np
import pandas as pd
import keras_nlp
from embetter.base import EmbetterBase


class KerasNLPEncoder(EmbetterBase):
    """
    Encoder that can numerically encode sentences.

    Arguments:
        name: name of model, see available options
        device: manually override cpu/gpu device, tries to grab gpu automatically when available
        quantize: turns on quantization
        num_threads: number of treads for pytorch to use, only affects when device=cpu

    The pre-trained model names that you could use can be found [here](https://keras.io/api/keras_nlp/models/).

    **Usage**:

    You can leverage the multiple backends from keras-core by setting the `KERAS_BACKEND` environment variable.

    ```python
    import os
    # Pick the right setting
    os.environ["KERAS_BACKEND"] = "jax"
    os.environ["KERAS_BACKEND"] = "torch"
    os.environ["KERAS_BACKEND"] = "tensorflow"
    ```

    Once this is set, the following code will automatically use the right backend.

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
        KerasNLPEncoder()
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

    def __init__(self, name="bert_tiny_en_uncased"):
        self.name = name
        self.backbone = keras_nlp.models.BertBackbone.from_preset(name)
        self.preprocessor = keras_nlp.models.BertPreprocessor.from_preset(name)

    def transform(self, X, y=None):
        """Transforms the text into a numeric representation."""
        if isinstance(X, pd.Series):
            X = X.to_numpy()
        out = self.backbone(self.preprocessor(X))["pooled_output"]

        # Depending on the backend, return numpy by calling right methods.
        if keras_nlp.src.backend.config.backend() == "torch":
            return out.detach().numpy()
        else:
            return np.asarray(out)
