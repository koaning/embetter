import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

__version__ = '0.0.1'


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
    def __init__(self, multi_output=False, size=32, epochs=50):
        self.size = size
        self.multi_output = multi_output
        self.epochs = epochs
        self.model = SimilarityModel(size=size)
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        self.binarizer = LabelBinarizer()
    
    def translate(self, X, y, classes=None):
        if not self.multi_output:
            if classes:
                self.binarizer.fit(classes)
            else:
                self.binarizer.fit(y)
            y = self.binarizer.transform(y)
        X1_out, X2_out, sim_out = [], [], []
        for i in range(X.shape[0]):
            x1_new = np.concatenate([X[i, :], np.zeros_like(y[i, :])], axis=0)
            # First we add all the positive cases
            for j in range(y.shape[1]):
                y_sim = y[i, j]
                if y_sim == 1:
                    yval = np.zeros(y.shape[1])
                    yval[j] = 1 
                    x2_new = np.concatenate([np.zeros_like(X[i, :]), yval], axis=0)
                    X1_out.append(x1_new)
                    X2_out.append(x2_new)
                    sim_out.append(y_sim)
            # Next we add only the negative cases
            for j in range(y.shape[1]):
                y_sim = y[i, j]
                if y_sim == 0:
                    yval = np.zeros(y.shape[1])
                    yval[j] = 1 
                    x2_new = np.concatenate([np.zeros_like(X[i, :]), yval], axis=0)
                    X1_out.append(x1_new)
                    X2_out.append(x2_new)
                    sim_out.append(y_sim)
            
        return np.array(X1_out), np.array(X2_out), np.array(sim_out)

    def _build_model(self):
        pass
