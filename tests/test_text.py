import pytest
import numpy as np

from embetter.text import SentenceEncoder, BytePairEncoder


def test_basic_sentence_encoder():
    """Check correct dimensions and repr for SentenceEncoder."""
    encoder = SentenceEncoder()
    # Embedding dim of underlying model
    output_dim = encoder.tfm._modules["1"].word_embedding_dimension
    test_sentences = [
        "This is a test sentence!",
        "And this is another one",
        "\rUnicode stuff: ♣️,♦️,❤️,♠️\n",
    ]
    output = encoder.fit_transform(test_sentences)
    assert isinstance(output, np.ndarray)
    assert output.shape == (len(test_sentences), output_dim)
    # scikit-learn configures repr dynamically from defined attributes.
    # To test correct implementation we should test if calling repr breaks.
    assert repr(encoder)


@pytest.mark.parametrize("setting", ["max", "mean", "both"])
def test_basic_bpemb(setting):
    """Check correct dimensions and repr for SentenceEncoder."""
    encoder = BytePairEncoder(lang="en", dim=50, agg=setting)
    # Embedding dim of underlying model
    test_sentences = [
        "This is a test sentence!",
        "And this is another one",
        "\rUnicode stuff: ♣️,♦️,❤️,♠️\n",
    ]
    output = encoder.fit_transform(test_sentences)
    assert isinstance(output, np.ndarray)
    assert output.shape == (len(test_sentences), 100 if setting == "both" else 50)
    # scikit-learn configures repr dynamically from defined attributes.
    # To test correct implementation we should test if calling repr breaks.
    assert repr(encoder)
