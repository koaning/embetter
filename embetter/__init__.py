import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

try:
    from importlib import metadata
except ImportError:  # for Python<3.8
    import importlib_metadata as metadata


__title__ = __name__
__version__ = metadata.version(__title__)


class Emb(tf.keras.Model):
    def __init__(self, size=32):
        super(Emb, self).__init__()
        self.dense1 = tf.keras.layers.Dense(size, activation=tf.nn.sigmoid)
        self.dense2 = tf.keras.layers.Dense(size)

    def call(self, x):
        return self.dense2(self.dense1(x))


class SimilarityModel(tf.keras.Model):
    def __init__(self, size=32):
        super(SimilarityModel, self).__init__()
        self.emb = Emb(size=size)

    def call(self, inputs):
        in1, in2 = inputs
        x1 = self.emb(in1)
        x2 = self.emb(in2)
        out = -tf.keras.losses.cosine_similarity(x1, x2, axis=1)
        return tf.keras.activations.sigmoid((out-0.5)*5)


class Embetter(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, multi_output=False, n_neg_samples=5, size=32, epochs=5, batch_size=512, verbose=1):
        self.size = size
        self.verbose = verbose
        self.n_neg_samples = n_neg_samples
        self.multi_output = multi_output
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = SimilarityModel(size=size)
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        self.binarizer = LabelBinarizer()

    def fit(self, X, y):
        X1, X2, y_sim = self.translate(X, y)
        self.model.fit([X1, X2], y_sim, epochs=self.epochs, verbose=self.verbose, batch_size=self.batch_size)
        return self
    
    def fit_sim(self, X1, X2, y):
        self.model.fit([X1, X2], y, epochs=self.epochs, verbose=self.verbose, batch_size=self.batch_size)
        return self
    
    def partial_fit_sim(self, X1, X2, y):
        self.model.fit([X1, X2], y, epochs=self.epochs, verbose=self.verbose, batch_size=self.batch_size)
        return self
    
    def embed(self, X):
        return self.model.emb(X)
