from pathlib import Path
import random
from collections import defaultdict
from itertools import chain, groupby
from typing import List, Union

import numpy as np
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from torch.utils.data import DataLoader
from dataclasses import dataclass

@dataclass
class Example:
    i1: int
    i2: int
    label: float

StrOrPath = Union[Path, str]


def generate_pairs_batch(labels, n_neg=3):
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
                [
                    indices
                    for label, indices in lookup.items()
                    if label != current_label
                ]
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


generate_pairs_batch(np.random.randint(0, 2, 100))

import torch
import torch.nn as nn 


class ContrastiveNetwork(nn.Module):
    """
    Network from Figure 1: https://arxiv.org/pdf/1908.10084.pdf. 
    """
    def __init__(self, shape_in, shape_out=1):
        super(ContrastiveNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(shape_in * 3, shape_out),
            nn.Sigmoid()
        )
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, input1, input2):
        concat = torch.cat((input1, input2, torch.abs(input1-input2)), 1)
        return self.fc(concat)


pairs = generate_pairs_batch("abcabcbcbababcbcacccaaaca")
X = np.random.random((20, 5))

X1 = np.zeros(shape=(len(pairs), X.shape[1]))
X2 = np.zeros(shape=(len(pairs), X.shape[1]))
for pair in pairs:
    X1[pair.i1] = X[pair.i1]
    X2[pair.i1] = X[pair.i2]
