import numpy as np
import pytest

from embetter.text import (
    SentenceEncoder,
    MatryoshkaEncoder,
    TextEncoder,
)

test_sentences = [
    "This is a test sentence!",
    "And this is another one",
    "\rUnicode stuff: ♣️,♦️,❤️,♠️\n",
]

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


def test_basic_text_encoder():
    """Check correct dimensions and repr for TextEncoder."""
    enc = TextEncoder()
    output = enc.fit_transform(test_sentences)
    assert isinstance(output, np.ndarray)
    assert repr(enc)
