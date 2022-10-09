import audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase
from hw_asr.augmentations.random_apply import RandomApply


class GaussianNoise(AugmentationBase):
    def __init__(self, p=0.2, *args, **kwargs):
        self._aug = RandomApply(audiomentations.AddGaussianNoise(*args, **kwargs), p)


    def __call__(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x, sample_rate=16000).squeeze(1)
