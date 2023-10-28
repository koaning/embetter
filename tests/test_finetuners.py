import pytest
from embetter.finetune import FeedForwardTuner, ContrastiveTuner, ContrastiveLearner
from sklearn.feature_extraction.text import CountVectorizer


texts = ["i am positive", "i am negative", "this is neutral"]
labels = ["pos", "neg", "neu"]


@pytest.mark.parametrize("finetuner", [ContrastiveTuner, FeedForwardTuner])
@pytest.mark.parametrize("hidden_dim", [25, 50, 75])
def test_finetuner_basics(finetuner, hidden_dim):
    """https://github.com/koaning/embetter/issues/38"""
    cv = CountVectorizer()
    X_tfm = cv.fit(texts).transform(texts)
    fft = finetuner(hidden_dim=hidden_dim)
    fft.fit(X_tfm.todense(), labels)
    assert fft.transform(X_tfm.todense()).shape[1] == hidden_dim
    assert repr(fft)


@pytest.mark.parametrize("learner", [ContrastiveLearner])
@pytest.mark.parametrize("shape_out", [25, 50, 75])
def test_learner_basics(learner, shape_out):
    cv = CountVectorizer()
    X_tfm = cv.fit(texts).transform(texts)
    fft = learner(shape_out=shape_out)
    X1, X2 = X_tfm.todense(), X_tfm.todense()
    fft.fit(X1, X2, [0, 1, 0])
    assert fft.transform(X_tfm.todense()).shape[1] == shape_out
    assert repr(fft)
