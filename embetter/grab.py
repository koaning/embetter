from PIL import Image 
from embetter.base import EmbetterBase


class ImageGrabber(EmbetterBase):
    """
    Component that can turn image paths into numpy arrays.
    """
    def __init__(self, convert='RGB') -> None:
        self.convert = convert
    
    def transform(self, X, y=None):
        """
        Turn a file path into numpy array containing pixel values.
        """
        return [Image.open(x).convert(self.convert) for x in X]
