from embetter.error import NotInstalled

try:
    from embetter.finetune._forward import FeedForwardTransformer
    from embetter.finetune._contrastive_tfm import ContrastiveTransformer
    from embetter.finetune._constrastive_learn import ContrastiveLearner
    from embetter.finetune._sbert_learn import SbertLearner
except ModuleNotFoundError:
    FeedForwardTransformer = NotInstalled("ForwardFinetuner", "pytorch")
    ContrastiveTransformer = NotInstalled("ContrastiveFinetuner", "pytorch")
    ContrastiveLearner = NotInstalled("ContrastiveLearner", "pytorch")
    SbertLearner = NotInstalled("SbertLearner", "pytorch")


__all__ = ["FeedForwardTransformer", "ContrastiveTransformer", "SbertLearner", "ContrastiveLearner"]
