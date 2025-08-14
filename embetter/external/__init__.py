from embetter.error import NotInstalled

try:
    from ._openai import OpenAIEncoder
except ModuleNotFoundError:
    OpenAIEncoder = NotInstalled("OpenAIEncoder", "openai")

try:
    from ._openai import AzureOpenAIEncoder
except ModuleNotFoundError:
    AzureOpenAIEncoder = NotInstalled("AzureOpenAIEncoder", "openai")

try:
    from ._cohere import CohereEncoder
except ModuleNotFoundError:
    CohereEncoder = NotInstalled("CohereEncoder", "cohere")


try:
    from ._ollama import OllamaEncoder
except ModuleNotFoundError:
    OllamaEncoder = NotInstalled("OllamaEncoder", "ollama")

__all__ = ["CohereEncoder", "OpenAIEncoder", "AzureOpenAIEncoder", "OllamaEncoder"]
