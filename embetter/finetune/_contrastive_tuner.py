from sklearn.base import BaseEstimator, TransformerMixin
import random
from collections import defaultdict
from itertools import chain, groupby

import numpy as np
import torch
from dataclasses import dataclass

from ._constrastive_learn import ContrastiveLearner


@dataclass
class Example:
    """Internal example class."""

    i1: int
    i2: int
    label: float


def generate_pairs_batch(labels, n_neg=3):
    """
    Copied with permission from Peter Baumgartners implementation
    https://github.com/pmbaumgartner/setfit
    """
    # 7x faster than original implementation on small data,
    # 14x faster on 10000 examples
    pairs = []
    lookup = defaultdict(list)
    single_example = {}
    indices = np.arange(len(labels))
    for label, grouper in groupby(
        ((s, lab) for s, lab in zip(indices, labels)), key=lambda x: x[1]
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


class ContrastiveTuner(BaseEstimator, TransformerMixin):
    """
    Run a contrastive network to finetune the embeddings towards a class.

    Arguments:
        hidden_dim: the dimension of the new learned representation
        n_neg: number of negative example pairs to sample per positive item
        n_epochs: number of epochs to use for training
        learning_rate: learning rate of the contrastive network
    """

    def __init__(self, hidden_dim=50, n_neg=3, epochs=20, learning_rate=0.001) -> None:
        self.learner = ContrastiveLearner(
            shape_out=hidden_dim,
            batch_size=256,
            learning_rate=learning_rate,
            epochs=epochs,
        )
        self.n_neg = n_neg
        self.hidden_dim = hidden_dim
        self.epochs = epochs
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

    def partial_fit(self, X, y, classes=None):
        """Fits the finetuner using the partial_fit API."""
        if not hasattr(self, "_classes"):
            if classes is None:
                raise ValueError("`classes` must be provided for partial_fit")
            self._classes = classes

        X_torch = torch.from_numpy(X).detach().float()

        X1, X2, out = self.generate_batch(X_torch, y=y)
        # TODO: change this, we should just generate numpy internally not cast all over
        self.learner.fit(np.array(X1), np.array(X2), np.array(out))

        return self

    def transform(self, X, y=None):
        """Transforms the data according to the sklearn api by using the hidden layer."""
        return self.learner.transform(X)
