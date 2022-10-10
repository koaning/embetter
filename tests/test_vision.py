import pytest
from embetter.vision import ImageLoader, ColorHistogramEncoder, TimmEncoder


@pytest.mark.parametrize("n_buckets", [5, 10, 25, 128])
def test_color_hist_resize(n_buckets):
    """Make sure we can resize and it fits"""
    X = ImageLoader().fit_transform(["tests/data/thiscatdoesnotexist.jpeg"])
    shape_out = ColorHistogramEncoder(n_buckets=n_buckets).fit_transform(X).shape
    shape_exp = (1, n_buckets * 3)
    assert shape_exp == shape_out


@pytest.mark.parametrize("encode_predictions,size", [(True, 1000), (False, 1280)])
def test_basic_timm(encode_predictions, size):
    """Super basic check for torch image model."""
    model = TimmEncoder("mobilenetv2_120d", encode_predictions=encode_predictions)
    X = ImageLoader().fit_transform(["tests/data/thiscatdoesnotexist.jpeg"])
    out = model.fit_transform(X)
    assert out.shape == (1, size)
