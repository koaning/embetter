from embetter.error import NotInstalled

try:
    from ._openai import OpenAIEncoder
except ModuleNotFoundError:
    OpenAIEncoder = NotInstalled("OpenAIEncoder", "openai")

try:
    from ._cohere import CohereEncoder
except ModuleNotFoundError:
    CohereEncoder = NotInstalled("CohereEncoder", "cohere")


__all__ = ["CohereEncoder", "OpenAIEncoder"]
