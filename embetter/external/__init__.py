from embetter.error import NotInstalled

from ._cohere import CohereEncoder

try:
    from ._openai import OpenAIEncoder
except ModuleNotFoundError:
    OpenAIEncoder = NotInstalled("OpenAIEncoder", "openai")


__all__ = ["CohereEncoder", "OpenAIEncoder"]
