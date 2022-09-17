from embetter.base import EmbetterBase


class ColumnGrabber(EmbetterBase):
    """
    Component that can grab a pandas column as a list.

    This can be useful when dealing with text encoders as these
    sometimes cannot deal with pandas columns.

    ### Arguments:
     - `colname`: the column name to grab from a dataframe

    ### Usage

    The most common way to use the ColumnGrabber is part of a pipeline.

    ```python
    import pandas as pd
    from sklearn.pipeline import make_pipeline

    from embetter.grab import ColumnGrabber
    from embetter.vision import ImageGrabber, ColorHistogram

    # Let's say we start we start with a csv file with filepaths
    data = {"filepaths":  ["tests/data/thiscatdoesnotexist.jpeg"]}
    df = pd.DataFrame(data)

    # Let's build a pipeline that grabs the column, turns it
    # into an image and embeds it.
    pipe = make_pipeline(
        ColumnGrabber("filepaths"),
        ImageGrabber(),
        ColorHistogram()
    )

    pipe.fit_transform(df)
    ```
    """

    def __init__(self, colname) -> None:
        self.colname = colname

    def transform(self, X, y=None):
        """
        Takes a column from pandas and returns it as a list.
        """
        return [x for x in X[self.colname]]
