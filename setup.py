import pathlib

from setuptools import find_packages, setup

base_packages = [
    "scikit-learn>=1.0.0",
    "pandas>=1.0.0",
    "diskcache>=5.6.1",
    "skops>=0.8.0",
    "model2vec",
]

sbert_pkgs = ["sentence-transformers>=2.2.2"]
sense2vec_pkgs = ["sense2vec==2.0.0"]
spacy_packages = ["spacy>=3.5.0"]

text_packages = sbert_pkgs + sense2vec_pkgs + spacy_packages

vision_packages = ["timm>=0.6.7"]

pytorch_packages = ["torch>=1.12.0"]

openai_packages = ["openai>=1.59.8"]

cohere_packages = ["cohere>=4.11.2"]

ollama_packages = ["ollama >= 0.5.3"]


docs_packages = [
    "mkdocs-material==9.6.9",
    "mkdocstrings==0.29.0",
    "mkdocstrings-python==1.16.0",
    "mktestdocs==0.2.4",
]

test_packages = [
    "interrogate>=1.5.0",
    "pytest>=4.0.2",
    "ruff",
    "pre-commit>=2.2.0",
    "mktestdocs==0.2.4",
    "datasets==2.8.0",
    "pyarrow==20.0.0",
    "matplotlib",
    "pytest-xdist",
]

all_packages = base_packages + text_packages + vision_packages + openai_packages
dev_packages = all_packages + docs_packages + test_packages


setup(
    name="embetter",
    version="0.8.0",
    author="Vincent D. Warmerdam",
    packages=find_packages(exclude=["notebooks", "docs", "datasets"]),
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
        "sbert": sbert_pkgs + base_packages,
        "spacy": spacy_packages + base_packages,
        "text": text_packages + base_packages,
        "vision": vision_packages + base_packages,
        "pytorch": pytorch_packages + base_packages,
        "openai": openai_packages + base_packages,
        "cohere": cohere_packages + base_packages,
        "ollama": ollama_packages + base_packages,
        "all": all_packages,
        "docs": docs_packages,
        "dev": dev_packages,
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
