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

StrOrPath = Union[Path, str]


def generate_sentence_pair_batch(
    sentences: List[str], labels: List[float]
) -> List[InputExample]:
    # 7x faster than original implementation on small data,
    # 14x faster on 10000 examples
    pairs = []
    sent_lookup = defaultdict(list)
    single_example = {}
    for label, grouper in groupby(
        ((s, l) for s, l in zip(sentences, labels)), key=lambda x: x[1]
    ):
        sent_lookup[label].extend(list(i[0] for i in grouper))
        single_example[label] = len(sent_lookup[label]) == 1
    neg_lookup = {}
    for current_label in sent_lookup:
        negative_options = list(
            chain.from_iterable(
                [
                    sentences
                    for label, sentences in sent_lookup.items()
                    if label != current_label
                ]
            )
        )
        neg_lookup[current_label] = negative_options

    for current_sentence, current_label in zip(sentences, labels):
        positive_pair = random.choice(sent_lookup[current_label])
        if not single_example[current_label]:
            # choosing itself as a matched pair seems wrong,
            # but we need to account for the case of 1 positive example
            # so as long as there's not a single positive example,
            # we'll reselect the other item in the pair until it's different
            while positive_pair == current_sentence:
                positive_pair = random.choice(sent_lookup[current_label])

        negative_pair = random.choice(neg_lookup[current_label])
        pairs.append(InputExample(texts=[current_sentence, positive_pair], label=1.0))
        pairs.append(InputExample(texts=[current_sentence, negative_pair], label=0.0))

    return pairs


def generate_multiple_sentence_pairs(
    sentences: List[str], labels: List[float], iter: int = 1
):
    all_pairs = []
    for _ in range(iter):
        all_pairs.extend(generate_sentence_pair_batch(sentences, labels))
    return all_pairs


class ContrastiveFinetuner(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model: str,
        data_iter: int = 5,
        train_iter: int = 1,
        batch_size: int = 16,
        show_progress_bar: bool = True,
        warmup_steps: int = 10,
        random_state: int = 1234,
    ):
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        self.random_state = random_state
        self.model = SentenceTransformer(model)
        self.loss = losses.CosineSimilarityLoss(self.model)
        self.fitted = False
        self.data_iter: int = data_iter,
        self.train_iter: int = train_iter,
        self.batch_size: int = batch_size,
        self.warmup_steps: int = warmup_steps,
        self.show_progress_bar: bool = show_progress_bar,

    def fit(
        self,
        X,
        y,
    ):
        train_examples = generate_multiple_sentence_pairs(X, y, self.data_iter)
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=self.batch_size,
            generator=torch.Generator(device=self.model.device),
        )
        self.model.fit(
            train_objectives=[(train_dataloader, self.loss)],
            epochs=self.train_iter,
            warmup_steps=self.warmup_steps,
            show_progress_bar=self.show_progress_bar,
        )
        self.fitted = True

    def transform(self, X, y=None):
        if not self.fitted:
            raise NotFittedError(
                "This SetFitClassifier instance is not fitted yet."
                " Call 'fit' with appropriate arguments before using this estimator."
            )
        return self.model.encode(X)
