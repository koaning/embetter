from embetter.error import NotInstalled

try:
    from embetter.finetune._forward import ForwardFinetuner
except ModuleNotFoundError:
    ForwardFinetuner = NotInstalled("ForwardFinetuner", "pytorch")


__all__ = ["ForwardFinetuner"]
