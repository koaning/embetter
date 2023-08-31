import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.base import TransformerMixin, ClassifierMixin


class DifferenceClassifier:
    """
    Classifier for similarity using encoders under the hood.

    It's similar to the scikit-learn models that you're used to, but it accepts
    two inputs `X1` and `X2` and tries to predict if they are similar. Effectively
    it's just a classifier on top of `diff(X1 - X2)`.

    Arguments:
        enc: scikit-learn compatbile encoder of the input data
        clf_head: the classifier to apply at the end

    Usage:

    ```python
    from embetter.model import DifferenceClassifier
    from embetter.text import SentenceEncoder

    mod = DifferenceClassifier(enc=SentenceEncoder())

    # Suppose this is input data
    texts1 = ["hello", "firehydrant", "greetings"]
    texts2 = ["no",    "yes",         "greeting"]

    # You will need to have some definition of "similar"
    similar = [0, 0, 1]

    # Train a model to detect similarity
    mod.fit(X1=texts1, X2=texts2, y=similar)
    mod.predict(X1=texts1, X2=texts2)

    # The classifier head is a scikit-learn model, which you could save
    # seperately if you like. The model can be accessed via:
    mod.clf_head
    ```
    """

    def __init__(self, enc: TransformerMixin, clf_head: ClassifierMixin = None):
        self.enc = enc
        self.clf_head = (
            LogisticRegression(class_weight="balanced") if not clf_head else clf_head
        )

    def _calc_feats(self, X1, X2):
        enc1 = self.enc.transform(X1)
        enc2 = self.enc.transform(X2)
        return np.abs(enc1 - enc2)

    def fit(self, X1, X2, y):
        self.clf_head.fit(self._calc_feats(X1, X2), y)
        return self

    def predict(self, X1, X2):
        return self.clf_head.predict(self._calc_feats(X1, X2))

    def predict_proba(self, X1, X2):
        return self.clf_head.predict_proba(self._calc_feats(X1, X2))
