import numpy as np
from embetter.base import EmbetterBase


class ColorHistogram(EmbetterBase):
    """
    Arguments:
    - n_buckets: number of buckets per color
    """

    def __init__(self, n_buckets=256):
        self.n_buckets = n_buckets

    def transform(self, X, y=None):
        """
        Takes a sequence of `PIL.Image` and returns a numpy array representing
        a color histogram for each.
        """
        output = np.zeros((len(X), self.n_buckets * 3))
        for i, x in enumerate(X):
            arr = np.array(x)
            output[i, :] = np.concatenate(
                [
                    np.histogram(
                        arr[:, :, 0].flatten(),
                        bins=np.linspace(0, 255, self.n_buckets + 1),
                    )[0],
                    np.histogram(
                        arr[:, :, 1].flatten(),
                        bins=np.linspace(0, 255, self.n_buckets + 1),
                    )[0],
                    np.histogram(
                        arr[:, :, 2].flatten(),
                        bins=np.linspace(0, 255, self.n_buckets + 1),
                    )[0],
                ]
            )
        return output
