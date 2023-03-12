from embetter.error import NotInstalled

try:
    from embetter.finetune._forward import ForwardFinetuner
    from embetter.finetune._contrastive import ContrastiveFinetuner
except ModuleNotFoundError:
    ForwardFinetuner = NotInstalled("ForwardFinetuner", "pytorch")
    ContrastiveFinetuner = NotInstalled("ContrastiveFinetuner", "pytorch")


__all__ = ["ForwardFinetuner", "ContrastiveFinetuner"]
