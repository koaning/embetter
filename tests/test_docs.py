from mktestdocs import check_md_file


def test_readme():
    """Readme needs to be accurate"""
    check_md_file(fpath="README.md")
