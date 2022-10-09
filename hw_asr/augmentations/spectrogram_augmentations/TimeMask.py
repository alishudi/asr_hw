from torch import Tensor, transforms

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.random_apply import RandomApply


class TimeMask(AugmentationBase):
    def __init__(self, pr=0.2, *args, **kwargs):
        self._aug = RandomApply(transforms.TimeMasking(*args, **kwargs), pr)

    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1)
