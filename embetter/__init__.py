from embetter.error import NotInstalled

try:
    from importlib import metadata
except ImportError:  # for Python<3.8
    import importlib_metadata as metadata


__title__ = __name__
__version__ = metadata.version(__title__)

try:
    from .text import SBERT
except ModuleNotFoundError:
    SBERT = NotInstalled("sentence-transformers", "sbert")
