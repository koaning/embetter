import numpy as np
from tqdm.auto import tqdm
from embetter.base import EmbetterBase


class ColorHistogramEncoder(EmbetterBase):
    """
    Encoder that generates an embedding based on the color histogram of the image.

    ![](https://raw.githubusercontent.com/koaning/embetter/main/docs/images/colorhistogram.png)

    Arguments:
        n_buckets: number of buckets per color
        show_progress_bar: Show a progress bar when encoding images

    **Usage**:

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

    # This pipeline can now encode each image in the dataframe
    pipe.fit_transform(df)
    ```
    """

    def __init__(self, n_buckets=256, show_progress_bar=False):
        self.n_buckets = n_buckets
        self.show_progress_bar = show_progress_bar

    def transform(self, X, y=None):
        """
        Takes a sequence of `PIL.Image` and returns a numpy array representing
        a color histogram for each.
        """
        output = np.zeros((len(X), self.n_buckets * 3))
        instances = tqdm(X, desc="ColorHistogramEncoder") if self.show_progress_bar else X
        for i, x in enumerate(instances):
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
