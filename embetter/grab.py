from embetter.base import EmbetterBase


class ColumnGrabber(EmbetterBase):
    """
    Component that can grab a pandas column as a list.

    ![](https://raw.githubusercontent.com/koaning/embetter/main/docs/images/columngrabber.png)

    This can be useful when dealing with text encoders as these
    sometimes cannot deal with pandas columns.

    Arguments:
        colname: the column name to grab from a dataframe

    **Usage**

    In essense, the `ColumnGrabber` really just selects a single column.

    ```python
    import pandas as pd
    from embetter.grab import ColumnGrabber

    # Let's say we start we start with a csv file with filepaths
    data = {"filepaths":  ["tests/data/thiscatdoesnotexist.jpeg"]}
    df = pd.DataFrame(data)

    # You can use the component in stand-alone fashion
    ColumnGrabber("filepaths").fit_transform(df)
    ```

    But the most common way to use the `ColumnGrabber` is part of a pipeline.

    ```python
    import pandas as pd
    from sklearn.pipeline import make_pipeline

    from embetter.grab import ColumnGrabber
    from embetter.vision import ImageLoader, ColorHistogramEncoder

    # Let's say we start we start with a csv file with filepaths
    data = {"filepaths":  ["tests/data/thiscatdoesnotexist.jpeg"]}
    df = pd.DataFrame(data)

    # You can use the component in stand-alone fashion
    ColumnGrabber("filepaths").fit_transform(df)

    # But let's build a pipeline that grabs the column, turns it
    # into an image and embeds it.
    pipe = make_pipeline(
        ColumnGrabber("filepaths"),
        ImageLoader(),
        ColorHistogramEncoder()
    )

    pipe.fit_transform(df)
    ```
    """

    def __init__(self, colname: str) -> None:
        self.colname = colname

    def transform(self, X, y=None):
        """
        Takes a column from pandas and returns it as a list.
        """
        return [x for x in X[self.colname]]


class KeyGrabber:
    """
    Effectively the same thing as the ColumnGrabber, except this is
    meant to work on generators of dictionaries instead of dataframes.
    """

    def __init__(self, colname: str) -> None:
        self.colname = colname

    def transform(self, X, y=None):
        """
        Takes a column from pandas and returns it as a list.
        """
        if isinstance(X, dict):
            return X[self.colname]
        return [x[self.colname] for x in X]
