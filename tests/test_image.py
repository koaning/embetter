import pytest
from embetter.vision import ImageGrabber, ColorHistogram


@pytest.mark.parametrize("n_buckets", [5, 10, 25, 128])
def test_color_hist_resize(n_buckets):
    """Make sure we can resize and it fits"""
    X = ImageGrabber().fit_transform(["tests/data/thiscatdoesnotexist.jpeg"])
    assert ColorHistogram(n_buckets=n_buckets).fit_transform(X).shape == (
        1,
        n_buckets * 3,
    )
