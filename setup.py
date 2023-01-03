import pathlib
from setuptools import setup, find_packages


base_packages = ["scikit-learn>=1.0.0", "pandas>=1.0.0"]

sentence_encoder_pkgs = ["sentence-transformers>=2.2.2"]
sense2vec_pkgs = ["sense2vec==2.0.0"]
text_packages = sentence_encoder_pkgs + sense2vec_pkgs

vision_packages = ["timm>=0.6.7"]

pytorch_packages = ["torch>=1.12.0"]

docs_packages = [
    "mkdocs==1.1",
    "mkdocs-material==4.6.3",
    "mkdocstrings==0.8.0",
    "mktestdocs==0.1.2",
]

test_packages = [
    "interrogate>=1.5.0",
    "flake8>=3.6.0",
    "pytest>=4.0.2",
    "black>=19.3b0",
    "pre-commit>=2.2.0",
    "mktestdocs==0.1.2",
    "datasets==2.8.0",
    "matplotlib==3.4.3"
]

all_packages = base_packages + text_packages + vision_packages
dev_packages = all_packages + docs_packages + test_packages


setup(
    name="embetter",
    version="0.2.3",
    author="Vincent D. Warmerdam",
    packages=find_packages(exclude=["notebooks", "docs"]),
    description="Just a bunch of useful embeddings to get started quickly.",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    license_files=("LICENSE"),
    url="https://koaning.github.io/embetter/",
    project_urls={
        "Documentation": "https://koaning.github.io/embetter/",
        "Source Code": "https://github.com/koaning/embetter/",
        "Issue Tracker": "https://github.com/koaning/embetter/issues",
    },
    install_requires=base_packages,
    extras_require={
        "sense2vec": sense2vec_pkgs + base_packages,
        "sentence-tfm": sentence_encoder_pkgs + base_packages,
        "text": text_packages + base_packages,
        "vision": vision_packages + base_packages,
        "pytorch": pytorch_packages + base_packages,
        "all": all_packages,
        "dev": dev_packages,
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
