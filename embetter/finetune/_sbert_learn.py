import numpy as np

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from torch.nn import CosineSimilarity


class SbertLearner:
    """
    A learner model that can finetune on pairs of data that leverages SBERT under the hood.

    It's similar to the scikit-learn models that you're used to, but it accepts
    two inputs `X1` and `X2` and tries to predict if they are similar.

    Arguments:
        sent_tfm: an instance of a `SentenceTransformer` that you'd like to finetune
        batch_size: the batch size during training
        epochs: the number of epochs to use while training
        warmup_steps: the number of warmup steps before training

    Usage:

    ```python
    from sentence_transformers import SentenceTransformer
    from embetter.finetune import SbertLearner
    import random

    sent_tfm = SentenceTransformer('all-MiniLM-L6-v2')
    learner = SbertLearner(sent_tfm)

    def sample_generator(examples, n_neg=3):
        # A generator that assumes examples to be a dictionary of the shape
        # {"text": "some text", "cats": {"label_a": True, "label_b": False}}
        # this is typically a function that's very custom to your use-case though
        labels = set()
        for ex in examples:
            for cat in ex['cats'].keys():
                if cat not in labels:
                    labels = labels.union([cat])
        for label in labels:
            pos_examples = [ex for ex in examples if label in ex['cats'] and ex['cats'][label] == 1]
            neg_examples = [ex for ex in examples if label in ex['cats'] and ex['cats'][label] == 0]
            for ex in pos_examples:
                sample = random.choice(pos_examples)
                yield (ex['text'], sample['text'], 1.0)
                for n in range(n_neg):
                    sample = random.choice(neg_examples)
                    yield (ex['text'], sample['text'], 0.0)

    learn_examples = sample_generator(examples, n_neg=3)
    X1, X2, y = zip(*learn_examples)

    # Learn a new representation
    learner.fit(X1, X2, y)

    # You now have an updated model that can create more "finetuned" embeddings
    learner.transform(X1)
    learner.transform(X2)
    ```

    After a learning is done training it can be used inside of a scikit-learn pipeline as you normally would.
    """

    def __init__(
        self,
        sent_tfm: SentenceTransformer,
        batch_size: int = 16,
        epochs: int = 1,
        warmup_steps: int = 100,
    ):
        self.sent_tfm = sent_tfm
        self.batch_size = batch_size
        self.epochs = epochs
        self.warmup_steps = warmup_steps

    def fit(self, X1, X2, y):
        """Finetune an Sbert model based on similarities between two sets of texts."""
        train_examples = [
            InputExample(texts=[x1, x2], label=float(lab))
            for x1, x2, lab in zip(X1, X2, y)
        ]
        data_loader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(self.sent_tfm)
        self.sent_tfm.fit(
            train_objectives=[(data_loader, train_loss)],
            epochs=self.epochs,
            warmup_steps=self.warmup_steps,
        )
        return self

    def transform(self, X, y=None):
        """Encode a single batch of Sbert inputs (usually texts)."""
        return self.sent_tfm.encode(X)

    def predict(self, X1, X2):
        """Predicts the cosine similarity."""
        emb1 = self.transform(X1)
        emb2 = self.transform(X2)
        return np.array(CosineSimilarity(dim=1)(emb1, emb2))

    def to_disk(self, path):
        """Save the finetuned Sbert model."""
        self.sent_tfm.save(path=path)
