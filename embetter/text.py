from sentence_transformers import SentenceTransformer
from embetter.base import BaseEstimator


class SBERT(BaseEstimator):
    def __init__(self, name='all-MiniLM-L6-v2'):
        self.name = name 
        self.tfm = SentenceTransformer(name)
    
    def transform(self, X, y=None):
        self.tfm.encode(X)
