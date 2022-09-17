import pandas as pd
from embetter.grab import ColumnGrabber


def test_grab_column():
    """Ensure that we can grab a text column."""
    data = [{"text": "hi", "foo": 1}, {"text": "yes", "foo": 2}]
    dataframe = pd.DataFrame(data)
    out = ColumnGrabber("text").fit_transform(dataframe)
    assert out == ["hi", "yes"]
