from embetter.error import NotInstalled

try:
    from ._openai import AzureOpenAIEncoder, OpenAIEncoder
except ModuleNotFoundError:
    OpenAIEncoder = NotInstalled("OpenAIEncoder", "openai")
    AzureOpenAIEncoder = NotInstalled("AzureOpenAIEncoder", "openai")

try:
    from ._cohere import CohereEncoder
except ModuleNotFoundError:
    CohereEncoder = NotInstalled("CohereEncoder", "cohere")


__all__ = ["CohereEncoder", "OpenAIEncoder", "AzureOpenAIEncoder"]
