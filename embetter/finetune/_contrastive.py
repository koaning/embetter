import random
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain, groupby
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class Example:
    """Internal example class."""

    i1: int
    i2: int
    label: float


def generate_pairs_batch(labels: List[str], n_neg=3):
    """
    Copied with permission from Peter Baumgartner implementation
    https://github.com/pmbaumgartner/setfit
    """
    # 7x faster than original implementation on small data,
    # 14x faster on 10000 examples
    pairs = []
    lookup = defaultdict(list)
    single_example = {}
    indices = np.arange(len(labels))
    for label, grouper in groupby(
        ((s, l) for s, l in zip(indices, labels)), key=lambda x: x[1]
    ):
        lookup[label].extend(list(i[0] for i in grouper))
        single_example[label] = len(lookup[label]) == 1
    neg_lookup = {}
    for current_label in lookup:
        negative_options = list(
            chain.from_iterable(
                [indices for label, indices in lookup.items() if label != current_label]
            )
        )
        neg_lookup[current_label] = negative_options

    for current_idx, current_label in zip(indices, labels):
        positive_pair = random.choice(lookup[current_label])
        if not single_example[current_label]:
            # choosing itself as a matched pair seems wrong,
            # but we need to account for the case of 1 positive example
            # so as long as there's not a single positive example,
            # we'll reselect the other item in the pair until it's different
            while positive_pair == current_idx:
                positive_pair = random.choice(lookup[current_label])
        pairs.append(Example(current_idx, positive_pair, 1))
        for i in range(n_neg):
            negative_pair = random.choice(neg_lookup[current_label])
            pairs.append(Example(current_idx, negative_pair, 0))

    return pairs


class ContrastiveNetwork(nn.Module):
    """
    Adapted from network from Figure 1: https://arxiv.org/pdf/1908.10084.pdf.
    """

    def __init__(self, shape_in, hidden_dim):
        super(ContrastiveNetwork, self).__init__()
        shape_out = 2
        self.emb = nn.Linear(shape_in, hidden_dim)
        # We multiply by three because we concat(u, v, |u - v|)
        # it's what the paper does https://github.com/koaning/embetter/issues/67
        self.fc = nn.Sequential(nn.Linear(hidden_dim * 3, shape_out), nn.Sigmoid())

    def embed(self, input_mat):
        """Return the learned embedding"""
        return self.emb(input_mat)

    def forward(self, input1, input2):
        """Feed forward"""
        emb_1 = self.embed(input1)
        emb_2 = self.embed(input2)
        out = torch.cat((emb_1, emb_2, torch.abs(emb_1 - emb_2)), dim=1)
        return self.fc(out)


class ContrastiveFinetuner(BaseEstimator, TransformerMixin):
    """
    Run a contrastive network to finetune the embeddings towards a class.

    Arguments:
        hidden_dim: the dimension of the new learned representation
        n_neg: number of negative example pairs to sample per positive item
        n_epochs: number of epochs to use for training
        learning_rate: learning rate of the contrastive network
    """

    def __init__(
        self, hidden_dim=50, n_neg=3, n_epochs=20, learning_rate=0.001
    ) -> None:
        self._criterion = None
        self._optimizer = None
        self._model = None
        self._classes = None
        self.hidden_dim = hidden_dim
        self.n_neg = n_neg
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate

    def fit(self, X, y):
        """Fits the finetuner."""
        return self.partial_fit(X, y, classes=np.unique(y))

    def generate_batch(self, X_torch, y):
        """Generate a batch of pytorch pairs used for finetuning"""
        pairs = generate_pairs_batch(y, n_neg=self.n_neg)
        X1 = torch.zeros(len(pairs), X_torch.shape[1])
        X2 = torch.zeros(len(pairs), X_torch.shape[1])
        labels = torch.tensor([ex.label for ex in pairs], dtype=torch.long)
        for i, pair in enumerate(pairs):
            X1[i] = X_torch[pair.i1]
            X2[i] = X_torch[pair.i2]
        return X1, X2, labels

    def partial_fit(self, X, y=None, classes=None):
        """Fits the finetuner using the partial_fit API."""
        if self._classes is None:
            if classes is None:
                raise ValueError("`classes` must be provided for partial_fit")
            self._classes = classes
        # Create a model if it does not exist yet.
        if self._model is None:
            self._model = ContrastiveNetwork(
                shape_in=X.shape[1], hidden_dim=self.hidden_dim
            )
            self._optimizer = torch.optim.Adam(
                self._model.parameters(), lr=self.learning_rate
            )
            self._criterion = nn.CrossEntropyLoss()

        X_torch = torch.from_numpy(X).detach().float()

        for _ in range(self.n_epochs):  # loop over the dataset multiple times
            X1, X2, out = self.generate_batch(X_torch, y=y)
            # zero the parameter gradients
            self._optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self._model(X1, X2)
            loss = self._criterion(outputs, out)
            loss.backward()
            self._optimizer.step()

        return self

    def transform(self, X, y=None):
        """Transforms the data according to the sklearn api by using the hidden layer."""
        Xt = torch.from_numpy(X).float().detach()
        return self._model.embed(Xt).detach().numpy()
