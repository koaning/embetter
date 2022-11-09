import numpy as np

from embetter.text import SentenceEncoder


def test_basic_sentence_encoder():
    encoder = SentenceEncoder()
    test_sentences = ["This is a test sentence!", "And this is another one", "\rUnicode stuff: ♣️,♦️,❤️,♠️\n"]
    output = encoder.fit_transform(test_sentences)
    assert isinstance(output, np.ndarray)
    assert output.shape == (len(test_sentences), 384)

