import pytest
from embetter.vision import ImageGrabber, ColorHistogramEncoder, TorchImageModels


@pytest.mark.parametrize("n_buckets", [5, 10, 25, 128])
def test_color_hist_resize(n_buckets):
    """Make sure we can resize and it fits"""
    X = ImageGrabber().fit_transform(["tests/data/thiscatdoesnotexist.jpeg"])
    shape_out = ColorHistogramEncoder(n_buckets=n_buckets).fit_transform(X).shape
    shape_exp = (1, n_buckets * 3)
    assert shape_exp == shape_out


def test_basic_timm():
    """Super basic check for torch image model."""
    model = TorchImageModels("mobilenetv2_120d")
    X = ImageGrabber().fit_transform(["tests/data/thiscatdoesnotexist.jpeg"])
    out = model.fit_transform(X)
    assert out.shape == (1, 1000)
