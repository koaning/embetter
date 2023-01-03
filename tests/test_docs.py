import pytest
from mktestdocs import check_md_file, check_docstring
from embetter.vision import ColorHistogramEncoder, TimmEncoder, ImageLoader
from embetter.text import Sense2VecEncoder, SentenceEncoder
from embetter.grab import ColumnGrabber


def test_readme():
    """Readme needs to be accurate"""
    check_md_file(fpath="README.md")


def test_finetune_docs():
    """Docs need to be accurate"""
    check_md_file(fpath="docs/finetuners.md", memory=True)


objects = [
    ColumnGrabber,
    SentenceEncoder,
    Sense2VecEncoder,
    ColorHistogramEncoder,
    TimmEncoder,
    ImageLoader,
]


@pytest.mark.parametrize("func", objects, ids=lambda d: d.__name__)
def test_docstring(func):
    """Check the docstrings of the components"""
    check_docstring(obj=func)
