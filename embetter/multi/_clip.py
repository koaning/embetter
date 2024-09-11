import pandas as pd
import torch
from torch.nn import Linear
from torch.quantization import quantize_dynamic
from sentence_transformers import SentenceTransformer as SBERT

from embetter.base import EmbetterBase


class ClipEncoder(EmbetterBase):
    """
    Clip model than can encode text and images.

    Under the hood it just wraps around the implementation of [sentence-transformers](https://sbert.net/docs/pretrained_models.html?highlight=clip)

    Arguments:
        name: name of model, see available options
        device: manually override cpu/mps/gpu device, tries to grab gpu or mps automatically when available
        quantize: turns on quantization
        num_threads: number of treads for pytorch to use, only affects when device=cpu

    The following model names should be supported:

    - `clip-ViT-B-32`
    - `clip-ViT-B-16`
    - `clip-ViT-B-14`
    - `clip-ViT-B-32-multilingual-v1`
    """

    def __init__(
        self, name="clip-ViT-B-32", device=None, quantize=False, num_threads=None
    ):
        if not device:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.name = name
        self.device = device
        self.tfm = SBERT(name, device=self.device)
        self.num_threads = num_threads
        self.quantize = quantize
        if quantize:
            self.tfm = quantize_dynamic(self.tfm, {Linear})
        if num_threads:
            if self.device.type == "cpu":
                torch.set_num_threads(num_threads)

    def transform(self, X, y=None):
        """Transforms the text into a numeric representation."""
        # Convert pd.Series objects to encode compatable
        if isinstance(X, pd.Series):
            X = X.to_numpy()

        return self.tfm.encode(X)
