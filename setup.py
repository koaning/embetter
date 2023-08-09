import pathlib

from setuptools import find_packages, setup

base_packages = ["scikit-learn>=1.0.0", "pandas>=1.0.0", "diskcache>=5.6.1"]

sentence_encoder_pkgs = ["sentence-transformers>=2.2.2"]
sense2vec_pkgs = ["sense2vec==2.0.0"]
bpemb_packages = ["bpemb>=0.3.3"]
spacy_packages = ["spacy>=3.5.0"]
gensim_packages = ["gensim>=4.3.1"]
keras_nlp_packages = ["keras-nlp>=0.6.0"]

text_packages = (
    sentence_encoder_pkgs
    + sense2vec_pkgs
    + bpemb_packages
    + gensim_packages
    + keras_nlp_packages
)

vision_packages = ["timm>=0.6.7"]

pytorch_packages = ["torch>=1.12.0"]

openai_packages = ["openai>=0.25.0"]

cohere_packages = ["cohere>=4.11.2"]


docs_packages = [
    "mkdocs==1.5.2",
    "mkdocs-material==9.1.21",
    "mkdocstrings==0.22.0",
    "mkdocstrings-python==1.3.0",
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
    "matplotlib==3.4.3",
]

all_packages = base_packages + text_packages + vision_packages + openai_packages
dev_packages = all_packages + docs_packages + test_packages


setup(
    name="embetter",
    version="0.5.1",
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
        "gensim": gensim_packages + base_packages,
        "sense2vec": sense2vec_pkgs + base_packages,
        "sentence-tfm": sentence_encoder_pkgs + base_packages,
        "spacy": spacy_packages + base_packages,
        "keras_nlp": keras_nlp_packages + base_packages,
        "bpemb": bpemb_packages + base_packages,
        "text": text_packages + base_packages,
        "vision": vision_packages + base_packages,
        "pytorch": pytorch_packages + base_packages,
        "openai": openai_packages + base_packages,
        "cohere": cohere_packages + base_packages,
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
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
