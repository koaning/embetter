import numpy as np
from PIL import Image

from embetter.base import EmbetterBase


class ImageLoader(EmbetterBase):
    """
    Component that can turn filepaths into a list of PIL.Image objects.

    Arguments:
        convert: Color [conversion setting](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert) from the Python image library.
        out: What kind of image output format to expect.

    **Usage**

    You can use the `ImageLoader` in standalone fashion.

    ```python
    from embetter.vision import ImageLoader

    filepath = "tests/data/thiscatdoesnotexist.jpeg"
    ImageLoader(convert="RGB").fit_transform([filepath])
    ```

    But it's more common to see it part of a pipeline.

    ```python
    import pandas as pd
    from sklearn.pipeline import make_pipeline

    from embetter.grab import ColumnGrabber
    from embetter.vision import ImageLoader, ColorHistogramEncoder

    # Let's say we start we start with a csv file with filepaths
    data = {"filepaths":  ["tests/data/thiscatdoesnotexist.jpeg"]}
    df = pd.DataFrame(data)

    # Let's build a pipeline that grabs the column, turns it
    # into an image and embeds it.
    pipe = make_pipeline(
        ColumnGrabber("filepaths"),
        ImageLoader(),
        ColorHistogramEncoder()
    )

    pipe.fit_transform(df)
    ```

    """

    def __init__(self, convert: str = "RGB", out: str = "pil") -> None:
        self.convert = convert
        self.out = out

    def fit(self, X, y=None):
        """
        Not actual "fitting" happens in this method, but it does check the input arguments
        per sklearn convention.
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
