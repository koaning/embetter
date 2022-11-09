import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config

import numpy as np
from tqdm.auto import tqdm
from embetter.base import EmbetterBase


class TimmEncoder(EmbetterBase):
    """
    Use a pretrained vision model from TorchVision to generate embeddings. Embeddings
    are provider via the lovely `timm` library.

    ![](https://raw.githubusercontent.com/koaning/embetter/main/docs/images/timm.png)

    You can find a list of available models [here](https://rwightman.github.io/pytorch-image-models/models/).

    Arguments:
        name: name of the model to use
        encode_predictions: output the predictions instead of the pooled embedding layer before
        show_progress_bar: Show a progress bar when processing images

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

    def __init__(self, name="mobilenetv3_large_100", encode_predictions=False, show_progress_bar=False):
        self.name = name
        self.encode_predictions = encode_predictions
        self.model = timm.create_model(name, pretrained=True, num_classes=0)
        if self.encode_predictions:
            self.model = timm.create_model(name, pretrained=True)
        self.config = resolve_data_config({}, model=self.model)
        self.transform_img = create_transform(**self.config)
        self.show_progress_bar = show_progress_bar

    def transform(self, X, y=None):
        """
        Transforms grabbed images into numeric representations.
        """
        batch = [self.transform_img(x).unsqueeze(0) for x in X]
        instances = tqdm(batch, desc="Encoding using Timm") if self.show_progress_bar else batch
        output = []
        for x in instances:
            output.append(self.model(x).squeeze(0).detach().numpy())
        return np.array(output)
