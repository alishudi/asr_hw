import audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class TimeStretch(AugmentationBase):
    def __init__(self,*args, **kwargs):
        self._aug = audiomentations.TimeStretch(*args, **kwargs)

    def __call__(self, data: Tensor):
        try:
            x = data.unsqueeze(1).numpy()
            return Tensor(self._aug(x, sample_rate=16000)).squeeze(1)
        except:
            x = data.numpy()
            return Tensor(self._aug(x, sample_rate=16000))
