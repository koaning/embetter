import numpy as np

from embetter.utils import calc_distances
from embetter.text import SentenceEncoder


def test_calc_distances():
    """Make sure that the aggregation works as expected"""
    text_in = ["hi there", "no", "what is this then"]

    dists1 = calc_distances(
        text_in, ["greetings", "something else"], SentenceEncoder(), aggregate=np.min
    )
    dists2 = calc_distances(
        text_in,
        ["greetings", "something unrelated"],
        SentenceEncoder(),
        aggregate=np.min,
    )
    assert np.isclose(dists1.min(), dists2.min())
