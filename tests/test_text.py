import pytest
import numpy as np

from spacy.vocab import Vocab
from spacy.language import Language

from embetter.text import SentenceEncoder, BytePairEncoder, spaCyEncoder
from embetter.utils import cached


test_sentences = [
    "This is a test sentence!",
    "And this is another one",
    "\rUnicode stuff: ♣️,♦️,❤️,♠️\n",
]


def test_basic_sentence_encoder():
    """Check correct dimensions and repr for SentenceEncoder."""
    encoder = SentenceEncoder()
    # Embedding dim of underlying model
    output_dim = encoder.tfm._modules["1"].word_embedding_dimension
    output = encoder.fit_transform(test_sentences)
    assert isinstance(output, np.ndarray)
    assert output.shape == (len(test_sentences), output_dim)
    # scikit-learn configures repr dynamically from defined attributes.
    # To test correct implementation we should test if calling repr breaks.
    assert repr(encoder)


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
