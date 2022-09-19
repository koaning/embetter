import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config

import numpy as np
from embetter.base import EmbetterBase


class TimmEncoder(EmbetterBase):
    """
    Use a pretrained vision model from TorchVision to generate embeddings. Embeddings
    are provider via the lovely `timm` library.

    ![](https://raw.githubusercontent.com/koaning/embetter/main/docs/images/timm.png)

    You can find a list of available models [here](https://rwightman.github.io/pytorch-image-models/models/).

    **Usage**:

    ```python
    import pandas as pd
    from sklearn.pipeline import make_pipeline

    from embetter.grab import ColumnGrabber
    from embetter.vision import ImageLoader, TimmEncoder

    # Let's say we start we start with a csv file with filepaths
    data = {"filepaths":  ["tests/data/thiscatdoesnotexist.jpeg"]}
    df = pd.DataFrame(data)

    # Let's build a pipeline that grabs the column, turns it
    # into an image and embeds it.
    pipe = make_pipeline(
        ColumnGrabber("filepaths"),
        ImageLoader(),
        TimmEncoder(name="mobilenetv3_large_100")
    )

    # This pipeline can now encode each image in the dataframe
    pipe.fit_transform(df)
    ```
    """

    def __init__(
        self,
        name="mobilenetv3_large_100",
    ):
        self.name = name
        self.model = timm.create_model(name, pretrained=True)
        self.config = resolve_data_config({}, model=self.model)
        self.transform_img = create_transform(**self.config)

    def transform(self, X, y=None):
        """
        Transforms grabbed images into numeric representations.
        """
        batch = [self.transform_img(x).unsqueeze(0) for x in X]
        return np.array([self.model(x).squeeze(0).detach().numpy() for x in batch])
