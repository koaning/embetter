import pytest
from mktestdocs import check_md_file

def test_readme():
    check_md_file(fpath="README.md")
