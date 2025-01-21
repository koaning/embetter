import tempfile

import numpy as np
import pytest
from gensim.models import Word2Vec
from gensim.utils import tokenize
from spacy.language import Language
from spacy.vocab import Vocab

from embetter.text import (
    BytePairEncoder,
    SentenceEncoder,
    GensimEncoder,
    spaCyEncoder,
    MatryoshkaEncoder,
)
from embetter.utils import cached

test_sentences = [
    "This is a test sentence!",
    "And this is another one",
    "\rUnicode stuff: ♣️,♦️,❤️,♠️\n",
]


@pytest.mark.parametrize("setting", ["max", "mean", "both"])
def test_word2vec(setting):
    """Check if one can train and use a very simple word embedding model."""
    vector_size = 25
    sentences = [list(tokenize(sent)) for sent in test_sentences]
    model = Word2Vec(
        sentences=sentences, vector_size=vector_size, window=3, min_count=1
    )
    encoder = GensimEncoder(model, agg=setting)
    output = encoder.fit_transform(test_sentences)
    assert isinstance(output, np.ndarray)
    out_dim = vector_size if setting != "both" else vector_size * 2
    assert output.shape == (len(test_sentences), out_dim)
    # This tests whether it can load the model from disk
    with tempfile.NamedTemporaryFile() as fp:
        model.save(fp)
        encoder = GensimEncoder(fp.name, agg=setting)
        encoder.transform(test_sentences)
    assert repr(encoder)


@pytest.mark.parametrize("encoder", [MatryoshkaEncoder, SentenceEncoder])
def test_basic_sentence_encoder(encoder):
    """Check correct dimensions and repr for SentenceEncoder."""
    enc = encoder()
    # Embedding dim of underlying model
    output_dim = enc.tfm._modules["1"].word_embedding_dimension
    output = enc.fit_transform(test_sentences)
    assert isinstance(output, np.ndarray)
    assert output.shape == (len(test_sentences), output_dim)
    # scikit-learn configures repr dynamically from defined attributes.
    # To test correct implementation we should test if calling repr breaks.
    assert repr(enc)


@pytest.mark.parametrize("setting", ["max", "mean", "both"])
def test_basic_bpemb(setting):
    """Check correct dimensions and repr for BytePairEncoder."""
    encoder = BytePairEncoder(lang="en", dim=50, agg=setting)
    # Embedding dim of underlying model
    output = encoder.fit_transform(test_sentences)
    assert isinstance(output, np.ndarray)
    assert output.shape == (len(test_sentences), 100 if setting == "both" else 50)
    # scikit-learn configures repr dynamically from defined attributes.
    # To test correct implementation we should test if calling repr breaks.
    assert repr(encoder)


@pytest.fixture()
def nlp():
    """Just a fixture with a lightweight spaCy lang"""
    vector_data = {
        "red": np.array([1.0, 0.0]),
        "green": np.array([0.5, 0.5]),
        "blue": np.array([0.0, 1.0]),
        "purple": np.array([0.0, 1.0]),
    }

    vocab = Vocab(strings=list(vector_data.keys()))
    for word, vector in vector_data.items():
        vocab.set_vector(word, vector)
    return Language(vocab=vocab)


@pytest.mark.parametrize("setting", ["max", "mean", "both"])
def test_basic_spacy(setting, nlp):
    """Check correct dimensions and repr for spaCyEncoder."""
    encoder = spaCyEncoder(nlp, agg=setting)
    # Embedding dim of underlying model
    output = encoder.fit_transform(test_sentences)
    assert isinstance(output, np.ndarray)
    assert output.shape == (len(test_sentences), 4 if setting == "both" else 2)
    # scikit-learn configures repr dynamically from defined attributes.
    # To test correct implementation we should test if calling repr breaks.
    assert repr(encoder)


def test_basic_spacy_cached(nlp, tmpdir):
    """Just an e2e test for the cache."""
    encoder = spaCyEncoder(nlp)
    output_before = encoder.transform(test_sentences)

    # Now we cache it
    encoder = cached(tmpdir, encoder)
    output_during = encoder.transform(test_sentences)

    encoder = cached(tmpdir, encoder)
    output_after = encoder.transform(test_sentences)
    assert (output_before == output_during).all()
    assert (output_during == output_after).all()
