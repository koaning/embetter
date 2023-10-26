import torch 
import numpy as np 

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from torch.nn import CosineSimilarity
from torch import nn


class ContrastiveNetwork(nn.Module):
    def __init__(self, shape_in, hidden_dim):
        super(ContrastiveNetwork, self).__init__()
        self.embed = nn.Linear(shape_in, hidden_dim)
        self.cos = nn.CosineSimilarity()

    def forward(self, input1, input2):
        """Feed forward"""
        emb_1 = self.embed(input1)
        emb_2 = self.embed(input2)
        return self.cos(emb_1, emb_2)


class ContrastiveLearner:
    """
    """

    def __init__(self, shape_out:int = 300, batch_size:int = 16, epochs: int=1):
        self.network_ = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.shape_out = shape_out

    def fit(self, X1, X2, y):
        """Finetune an Sbert model based on similarities between two sets of texts."""
        self.network_ = ContrastiveNetwork(shape_in=X1.shape[0], shape_out=self.shape_out)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.network_.parameters(), lr=self.learning_rate
        )

        X1_torch = torch.from_numpy(X1).detach().float()
        X2_torch = torch.from_numpy(X1).detach().float()
        y_torch = torch.from_numpy(y).detach().float()

        for _ in range(self.n_epochs):  # loop over the dataset multiple times
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            cos_sim = self.network_(X1_torch, X2_torch)
            loss = criterion(cos_sim, y_torch)
            loss.backward()
            optimizer.step()

        return self

    def transform(self, X, y=None):
        """Encode a single batch of inputs."""
        X_torch = torch.from_numpy(X).detach().float()
        return np.array(self.network_.embed(X_torch))

    def predict(self, X1, X2):
        """Predicts the cosine similarity."""
        emb1 = self.transform(X1)
        emb2 = self.transform(X2)
        return np.array(CosineSimilarity(dim=1)(emb1, emb2))

    def to_disk(self, path):
        """Save the finetuned Sbert model."""
        self.sent_tfm.save(path=path)
