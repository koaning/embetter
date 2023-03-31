import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

from embetter.base import EmbetterBase


class UniversalSentenceEncoder(EmbetterBase):
    """
    Encoder that can numerically encode sentences.

    Arguments:
        name: name of model, see available options
        device: manually override cpu/gpu device, tries to grab gpu automatically when available

    The following model names should be supported:

    - `universal-sentence-encoder/4`
    - `universal-sentence-encoder-large/5`
    - `universal-sentence-encoder-multilingual/3`
    - `universal-sentence-encoder-multilingual-large/3`

    You can find the more options, and information, on the [Universal Sentence Encoder hub tutorial](.https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder#evaluation_sts_semantic_textual_similarity_benchmark)

    **Usage**:

    ```python
    import pandas as pd
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression

    from embetter.grab import ColumnGrabber
    from embetter.text import

    # Let's suppose this is the input dataframe
    dataf = pd.DataFrame({
        "text": ["positive sentiment", "super negative"],
        "label_col": ["pos", "neg"]
    })

    # This pipeline grabs the `text` column from a dataframe
    # which then get fed into Tensorflows' universal-sentence-ecnoder-lite/2.
    text_emb_pipeline = make_pipeline(
        ColumnGrabber("text"),
        UniversalSentenceEncoder('universal-sentence-encoder-lite/2')
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

    def __init__(self, name="universal-sentence-encoder/4", device=None):
        if not device:
            device = "/GPU:0" if self.__is_gpu_available() else "/CPU:0"
        self.model_url = f"https://tfhub.dev/google/{name}"
        self.device = device
        with tf.device(self.device):
            self.tfm = hub.load(self.model_url)

    def __repr__(self) -> str:
        return f"UniversalSentenceEncoder(model_url={self.model_url})"

    def __is_gpu_available(self):
        gpu_devices = tf.config.list_physical_devices("GPU")
        if len(gpu_devices) > 0:
            return True
        else:
            return False

    def transform(self, X, y=None):
        """Transforms the text into a numeric representation."""
        # Convert pd.Series objects to encode compatable
        if isinstance(X, pd.Series):
            X = X.to_numpy()

        with tf.device(self.device):
            sentence_embeddings = self.tfm(X)

        return sentence_embeddings.numpy()
