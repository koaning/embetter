import pytest
from embetter.finetune import ForwardFinetuner, ContrastiveFinetuner
from sklearn.feature_extraction.text import CountVectorizer


texts = ["i am positive", "i am negative", "this is neutral"]
labels = ["pos", "neg", "neu"]


@pytest.mark.parametrize("finetuner", [ContrastiveFinetuner, ForwardFinetuner])
def test_finetuner_basics(finetuner):
    """https://github.com/koaning/embetter/issues/38"""
    cv = CountVectorizer()
    X_tfm = cv.fit(texts).transform(texts)
    fft = finetuner(hidden_dim=75)
    fft.fit(X_tfm.todense(), labels)
    assert fft.transform(X_tfm.todense()).shape[1] == 75
