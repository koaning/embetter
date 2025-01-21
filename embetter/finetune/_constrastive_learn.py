import torch
import numpy as np

from torch.nn import CosineSimilarity
from torch import nn


class ContrastiveNetwork(nn.Module):
    def __init__(self, shape_in, hidden_dim):
        super(ContrastiveNetwork, self).__init__()
        self.embed1 = nn.Linear(shape_in, hidden_dim)
        self.embed2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.cos = nn.CosineSimilarity()

    def forward(self, input1, input2):
        """Feed forward."""
        emb_1 = self.embed2(self.act(self.embed1(input1)))
        emb_2 = self.embed2(self.act(self.embed1(input2)))
        return self.cos(emb_1, emb_2)

    def embed(self, X):
        return self.embed2(self.act(self.embed1(X)))


class ContrastiveLearner:
    """
    A learner model that can finetune on pairs of data on top of numeric embeddings.

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
    from embetter.finetune import ContrastiveLearner
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
        shape_out: int = 300,
        batch_size: int = 16,
        epochs: int = 1,
        learning_rate=2e-05,
    ):
        self.learning_rate = learning_rate
        self.network_ = None
        self.batch_size = batch_size
        self.epochs = epochs
        self.shape_out = shape_out

    def fit(self, X1, X2, y):
        """Finetune an Sbert model based on similarities between two sets of texts."""
        self.network_ = ContrastiveNetwork(
            shape_in=X1.shape[1], hidden_dim=self.shape_out
        )
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.network_.parameters(), lr=self.learning_rate)

        X1_torch = torch.from_numpy(X1).detach().float()
        X2_torch = torch.from_numpy(X2).detach().float()
        y_torch = torch.from_numpy(np.array(y)).detach().float()

        dataset = torch.utils.data.TensorDataset(X1_torch, X2_torch, y_torch)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        for _ in range(self.epochs):  # loop over the dataset multiple times
            for batch_X1, batch_X2, batch_y in dataloader:
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                cos_sim = self.network_(batch_X1, batch_X2)
                loss = criterion(cos_sim, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def transform(self, X, y=None):
        """Encode a single batch of inputs."""
        X_torch = torch.from_numpy(X).detach().float()
        return self.network_.embed(X_torch).detach().numpy()

    def predict(self, X1, X2):
        """Predicts the cosine similarity."""
        emb1 = self.transform(X1)
        emb2 = self.transform(X2)
        return np.array(CosineSimilarity()(emb1, emb2))

    def to_disk(self, path):
        """Save the finetuned Sbert model."""
        self.sent_tfm.save(path=path)
