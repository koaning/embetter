from itertools import islice
from typing import Callable, Iterable

import numpy as np
from diskcache import Cache
from sklearn.base import BaseEstimator
from sklearn.metrics import pairwise_distances


def cached(name: str, pipeline: BaseEstimator):
    """
    Uses a [diskcache](https://grantjenks.com/docs/diskcache/tutorial.html) in
    an attempt to fetch precalculated embeddings from disk instead of inferring them.
    This can save on compute, but also cloud credits, depending on the backend
    that you're using to generate embeddings.

    Be mindful of what does in to the encoder that you choose. It's preferable to give it
    text as opposed to numpy arrays. Also note that the first time that you'll run this
    it will take more time due to the overhead of writing into the cache.

    Arguments:
        name: the name of the local folder to represent the disk cache
        pipeline: the pipeline that you want to cache

    Usage:
    ```python
    from embetter.text import SentenceEncoder
    from embetter.utils import cached

    encoder = cached("sentence-enc", SentenceEncoder('all-MiniLM-L6-v2'))

    examples = [f"this is a pretty long text, which is more expensive {i}" for i in range(10_000)]

    # This might be a bit slow ~17.2s on our machine
    encoder.transform(examples)

    # This should be quicker ~4.71s on our machine
    encoder.transform(examples)
    ```

    Note that you're also able to fetch the precalculated embeddings directly via:

    ```python
    from diskcache import Cache

    # Make sure that you use the same name as in `cached`
    cache = Cache("sentence-enc")
    # Use a string as a key, if it's precalculated you'll get an array back.
    cache["this is a pretty long text, which is more expensive 0"]
    ```
    """
    cache = Cache(name)

    def run_cached(method: Callable):
        def wrapped(X, y=None):
            results = {i: cache[x] if x in cache else "TODO" for i, x in enumerate(X)}
            text_todo = [X[i] for i, x in results.items() if str(x) == "TODO"]
            i_todo = [i for i, x in results.items() if str(x) == "TODO"]
            out = method(text_todo)
            with Cache(cache.directory) as reference:
                for i, text, x_tfm in zip(i_todo, text_todo, out):
                    results[i] = x_tfm
                    reference.set(text, x_tfm)
            return np.array([arr for i, arr in results.items()])

        return wrapped

    pipeline.transform = run_cached(pipeline.transform)

    return pipeline


def batched(iterable: Iterable, n: int = 64):
    """
    Takes an iterable and turns it into a batched iterable.

    Arguments:
        iterable: the input stream
        n: the batch size
    """
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    for batch in tuple(islice(it, n)):
        yield batch


def calc_distances(
    inputs,
    anchors,
    pipeline,
    anchor_pipeline=None,
    metric="cosine",
    aggregate=np.max,
    n_jobs=None,
):
    """
    Shortcut to compare a sequence of inputs to a set of anchors.

    The available metrics are: `cityblock`,`cosine`,`euclidean`,`haversine`,`l1`,`l2`,`manhattan` and `nan_euclidean`.

    You can read a verbose description of the metrics [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html#sklearn.metrics.pairwise.distance_metrics).

    Arguments:
        inputs: sequence of inputs to calculate scores for
        anchors: set/list of anchors to compare against
        pipeline: the pipeline to use to calculate the embeddings
        anchor_pipeline: the pipeline to apply to the anchors, meant to be used if the anchors should use a different pipeline
        metric: the distance metric to use
        aggregate: you'll want to aggregate the distances to the different anchors down to a single metric, numpy functions that offer axis=1, like `np.max` and `np.mean`, can be used
        n_jobs: set to -1 to use all cores for calculation
    """
    X_input = pipeline.transform(inputs)
    if anchor_pipeline:
        X_anchors = anchor_pipeline.transform(anchors)
    else:
        X_anchors = pipeline.transform(anchors)

    X_dist = pairwise_distances(X_input, X_anchors, metric=metric, n_jobs=n_jobs)
    return aggregate(X_dist, axis=1)
