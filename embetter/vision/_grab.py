import numpy as np
from PIL import Image
from embetter.base import EmbetterBase


class ImageGrabber(EmbetterBase):
    """
    Component that can turn image paths into numpy arrays.
    """

    def __init__(self, convert="RGB", out="pil") -> None:
        self.convert = convert
        self.out = out

    def fit(self, X, y=None):
        """
        Checks if input params are good
        """
        if self.out not in ["pil", "numpy"]:
            raise ValueError(
                f"Output format parameter out={self.out} must be either pil/numpy."
            )
        return self

    def transform(self, X, y=None):
        """
        Turn a file path into numpy array containing pixel values.
        """
        if self.out == "pil":
            return [Image.open(x).convert(self.convert) for x in X]
        if self.out == "numpy":
            return np.array([np.array(Image.open(x).convert(self.convert)) for x in X])
