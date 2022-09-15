import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config

import numpy as np
from embetter.base import EmbetterBase


class TimmEncoder(EmbetterBase):
    """
    Use a pretrained vision model from TorchVision to generate embeddings. Embeddings
    are provider via the lovely `timm` library.

    You can find a list of available models here:
    https://rwightman.github.io/pytorch-image-models/models/

    Usage:

    ```python
    from embetter import Timm

    # MobileNet
    Timm("mobilenetv2_120d")
    ```
    """

    def __init__(
        self,
        name,
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
