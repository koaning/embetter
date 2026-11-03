import os
from itertools import chain

import numpy as np
import pandas as pd
import pytest
from diskcache_community import Cache, Disk

from embetter.utils import batched, cached, calc_distances
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


class CountingEncoder:
    """Stand-in for an embedding model that remembers which texts it had to encode."""

    def __init__(self):
        self.seen = []

    def transform(self, X, y=None):
        self.seen.extend(X)
        return np.array([[len(x), 0.5] for x in X], dtype=np.float32)


class PlantedEntry:
    """Pickles to a call of os.mkdir, which makes a folder when it gets unpickled."""

    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (os.mkdir, (self.path,))


def test_cached_reuses_embeddings(tmp_path):
    """Texts seen before come from the cache, also when passed as a numpy array."""
    encoder = cached(str(tmp_path / "enc"), CountingEncoder())
    first = encoder.transform(["hello", "world"])
    again = encoder.transform(np.array(["hello", "world", "again"]))
    assert encoder.seen == ["hello", "world", "again"]
    assert again.dtype == np.float32
    assert np.array_equal(again[:2], first)


def test_cached_ignores_planted_pickle(tmp_path):
    """An entry that something else wrote as a pickle is encoded again, its code never runs."""
    folder, marker = str(tmp_path / "enc"), tmp_path / "planted-code-ran"
    encoder = cached(folder, CountingEncoder())
    first = encoder.transform(["hello", "world"])

    with Cache(folder, disk=Disk) as other_writer:
        other_writer.set("hello", PlantedEntry(str(marker)))

    assert np.array_equal(encoder.transform(["hello", "world"]), first)
    assert not marker.exists()
    assert encoder.seen == ["hello", "world", "hello"]


def test_cached_keeps_texts_and_embeddings_together_for_a_shuffled_series(tmp_path):
    """A Series whose index isn't 0..n-1 stores each embedding under its own text."""
    texts = ["apple", "banana", "cherry!", "dates"]
    expected = CountingEncoder().transform(texts)
    shuffled = pd.Series(texts, index=[3, 2, 1, 0])

    encoder = cached(str(tmp_path / "enc"), CountingEncoder())
    assert np.array_equal(encoder.transform(shuffled), expected)
    assert np.array_equal(encoder.transform(texts), expected)
    assert encoder.seen == texts
