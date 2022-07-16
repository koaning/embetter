import torch 
from embetter.base import EmbetterBase


class TorchVision(EmbetterBase):
    """
    Use a pretrained vision model from TorchVision to generate embeddings.

    You can find a list of available models here:
    https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights
    """
    def __init__(self, repo_or_dir="pytorch/vision", model="resnet50", weights_name="ResNet50_Weights.IMAGENET1K_V2"):
        self.weights = torch.hub.load(repo_or_dir, "get_weight", name=weights_name)
        self.model = torch.hub.load(repo_or_dir, model, weights=self.weights)
        self.preprocess = self.weights.transforms()

    def transform(self, X, y=None):
        """
        Transforms grabbed images into numeric representations.
        """
        return self.model(X).squeeze(0)
