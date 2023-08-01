import keras_nlp 
from embetter.base import EmbetterBase


class KerasNLPEncoder(EmbetterBase):
    def __init__(self, name="bert_tiny_en_uncased"):
        self.tokenizer = keras_nlp.models.BertTokenizer.from_preset(name)
        self.preprocessor = keras_nlp.models.BertPreprocessor.from_preset(name)
    
    def transform(self, X, y=None):
        return self.backbone(self.preprocessor(X))['pooled_output'].numpy()

