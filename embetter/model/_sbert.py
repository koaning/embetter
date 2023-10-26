from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from torch.nn import CosineSimilarity

class SbertLearner:
    """
    A learner model that can finetune on pairs of data.

    It's similar to the scikit-learn models that you're used to, but it accepts
    two inputs `X1` and `X2` and tries to predict if they are similar. 

    Arguments:
        sent_tfm: an instance of a `SentenceTransformer` that you'd like to finetune
        batch_size: the batch size during training
        epochs: the number of epochs to use while training
        warmup_steps: the number of warmup steps before training

    Usage:

    ```python
    ```
    """

    def __init__(self, sent_tfm: SentenceTransformer, batch_size:int = 16, epochs: int=1, warmup_steps: int=100):
        self.sent_tfm = sent_tfm
        self.batch_size = batch_size
        self.epochs = epochs
        self.warmup_steps = warmup_steps

    def fit(self, X1, X2, y):
        """Finetune an Sbert model based on similarities between two sets of texts."""
        train_examples = [InputExample(texts=[x1, x2], label=lab) for x1, x2, lab in zip(X1, X2, y)]
        data_loader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(self.sent_tfm)
        self.sent_tfm.fit(train_objectives=(data_loader, train_loss), epochs=self.epochs, warmup_steps=self.warmup_steps)
        return self

    def transform(self, X, y=None):
        """Encode a single batch of Sbert inputs (usually texts)."""
        return self.sent_tfm.encode(X)

    def predict(self, X1, X2):
        """Predicts the cosine similarity."""
        emb1 = self.transform(X1)
        emb2 = self.transform(X2)
        return CosineSimilarity(dim=1)(emb1, emb2)

    def to_disk(self, path):
        """Save the finetuned Sbert model."""
        self.sent_tfm.save(path=path)
