import numpy as np 
import torch 
import torch.nn as nn 
from sklearn.base import BaseEstimator, TransformerMixin

class FeedForwardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardModel, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(self.embed(x)))
    
    def embed(self, x):
        return self.relu(self.hidden(x))


class ForwardFinetuner(BaseEstimator, TransformerMixin):
    def __init__(self, hidden_dim=50, n_epochs=10, learning_rate=0.001) -> None:
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate

    def fit(self, X, y):
        for _ in range(self.n_epochs):
            self.partial_fit(X, y, classes=np.unique(y))
        return self
    
    def partial_fit(self, X, y, classes=None):
        if not self._classes:
            self._classes = classes
        # Create a model if it does not exist yet. 
        if not self._model:
            self._model = FeedForwardModel(X.shape[1], self.hidden_dim, len(self._classes))
            self._optimizer = torch.optim.SGD(self._model.parameters(), lr=self.learning_rate)
            self._criterion = nn.CrossEntropyLoss()

        self._optimizer.zero_grad()
        out = self._model(X)
        loss = self._criterion(out, y)
        loss.backward()
        self._optimizer.step()

        return self

    def transform(self, X, y=None):
        return self._model.embed(X).detach().numpy()
