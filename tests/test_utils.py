from itertools import chain

import numpy as np
import pytest

from embetter.utils import batched, calc_distances
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


def test_batched_groups_into_tuples():
    """The utility should yield tuples, the last one may be shorter."""
    assert list(batched(range(10), n=3)) == [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)]


@pytest.mark.parametrize("n", [1, 3, 64, 100])
def test_batched_keeps_all_items(n):
    """Chaining the batches back together must reproduce the input."""
    items = list(range(250))
    assert list(chain.from_iterable(batched(items, n=n))) == items


def test_batched_on_empty_input():
    """An empty iterable should not yield any batch."""
    assert list(batched([], n=3)) == []


def test_batched_with_n_larger_than_input():
    """A single batch comes out when n exceeds the number of items."""
    assert list(batched(range(3), n=10)) == [(0, 1, 2)]


def test_batched_on_generator():
    """The utility should also accept iterators, not just sequences."""
    assert list(batched((i for i in range(5)), n=2)) == [(0, 1), (2, 3), (4,)]


def test_batched_rejects_small_n():
    """A batch size below one makes no sense."""
    with pytest.raises(ValueError):
        list(batched(range(5), n=0))
