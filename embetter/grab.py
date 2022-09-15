from embetter.base import EmbetterBase


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
