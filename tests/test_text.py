import numpy as np

from embetter.text import SentenceEncoder


def test_basic_sentence_encoder():
    encoder = SentenceEncoder()
    output_dim = encoder.tfm._modules['1'].word_embedding_dimension
    test_sentences = ["This is a test sentence!", "And this is another one", "\rUnicode stuff: ♣️,♦️,❤️,♠️\n"]
    output = encoder.fit_transform(test_sentences)
    assert isinstance(output, np.ndarray)
    assert output.shape == (len(test_sentences), output_dim)

