from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

import itertools as it
from skops.io import dump, load


def learn_lite_text_embeddings(text_stream, dim=300, lite=True, path=None, **kwargs):
    """
    Function that can train a TF/iDF model followed by SVD to generate dense text representations.

    Arguments:
        path: path where model is saved

    This function can be used to load a model that's saved with `featherbed_textrepr`.

    **Usage**:

    You can leverage the multiple backends from keras-core by setting the `KERAS_BACKEND` environment variable.

    ```python
    from embetter.text import learn_lite_text_embeddings

    # Save a variable that contains the scikit-learn pipeline, but also store on disk.
    enc = learn_lite_text_embeddings(generator_of_strings, path="folder/embeddings.skops")
    ```
    """
    # Make two streams, keep memory footprint low
    stream1, stream2 = it.tee(text_stream)

    # Tf/Idf vectorizer can accept generators!
    tfidf = TfidfVectorizer(**kwargs).fit(stream1)
    X = tfidf.transform(stream2)
    if lite:
        # This makes a pretty big difference
        tfidf.idf_ = tfidf.idf_.astype("float16")

    # Turn the representation into floats
    svd = TruncatedSVD(n_components=dim, **kwargs).fit(X)

    # This makes it much more lightweight to save
    if lite:
        svd.components_ = svd.components_.astype("float16")
    pipe = make_pipeline(tfidf, svd)
    if path:
        # This makes a pretty big difference
        dump(pipe, path)
    return pipe


def LiteTextEncoder(path):
    """
    Function that looks like class so that it fits the API.

    Arguments:
        path: path where model is saved

    This function can be used to load a model that's saved with `featherbed_textrepr`.

    **Usage**:

    You can leverage the multiple backends from keras-core by setting the `KERAS_BACKEND` environment variable.

    ```python
    from embetter.text import learn_lite_text_embeddings, LiteTextEncoder

    learn_lite_text_embeddings(generator_of_strings, path="folder/embeddings.skops")

    enc = LiteTextEncoder(path="folder/embeddings.skops")
    enc.transform(["encode this examples", "and this one"])
    ```
    """
    return load(path, trusted=True)
