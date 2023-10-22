import pytest
from mktestdocs import check_md_file, check_docstring
from embetter.vision import ColorHistogramEncoder, TimmEncoder, ImageLoader
from embetter.text import SentenceEncoder, BytePairEncoder
from embetter.grab import ColumnGrabber
from embetter.model import DifferenceClassifier


def test_readme():
    """Readme needs to be accurate"""
    check_md_file(fpath="README.md")


# def test_finetune_docs():
#     """Docs need to be accurate"""
#     check_md_file(fpath="docs/finetuners.md", memory=True)


# I'm not testing spaCy, sense2vec because those docs would require
# us to download `en_core_web_md` on every CI. Which is too heavy.
objects = [
    ColumnGrabber,
    SentenceEncoder,
    ColorHistogramEncoder,
    TimmEncoder,
    ImageLoader,
    BytePairEncoder,
    DifferenceClassifier,
]


@pytest.mark.parametrize("func", objects, ids=lambda d: d.__name__)
def test_docstring(func):
    """Check the docstrings of the components"""
    check_docstring(obj=func)
