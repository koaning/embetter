from PIL import Image
from embetter.base import EmbetterBase


class ImageGrabber(EmbetterBase):
    """
    Component that can turn image paths into numpy arrays.
    """

    def __init__(self, convert="RGB") -> None:
        self.convert = convert

    def transform(self, X, y=None):
        """
        Turn a file path into numpy array containing pixel values.
        """
        return [Image.open(x).convert(self.convert) for x in X]


class ColumnGrabber(EmbetterBase):
    """
    Component that can grab a pandas column as a list.

    This can be useful when dealing with text encoders as these
    sometimes cannot deal with pandas columns.
    """

    def __init__(self, colname) -> None:
        self.colname = colname

    def transform(self, X, y=None):
        """
        Takes a column from pandas and returns it as a list.
        """
        return [x for x in X[self.colname]]
