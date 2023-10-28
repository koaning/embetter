import pytest
from embetter.finetune import FeedForwardTransformer, ContrastiveTransformer
from sklearn.feature_extraction.text import CountVectorizer


texts = ["i am positive", "i am negative", "this is neutral"]
labels = ["pos", "neg", "neu"]


@pytest.mark.parametrize("finetuner", [ContrastiveTransformer, FeedForwardTransformer])
@pytest.mark.parametrize("hidden_dim", [25, 50, 75])
def test_finetuner_basics(finetuner, hidden_dim):
    """https://github.com/koaning/embetter/issues/38"""
    cv = CountVectorizer()
    X_tfm = cv.fit(texts).transform(texts)
    fft = finetuner(hidden_dim=hidden_dim)
    fft.fit(X_tfm.todense(), labels)
    assert fft.transform(X_tfm.todense()).shape[1] == hidden_dim
