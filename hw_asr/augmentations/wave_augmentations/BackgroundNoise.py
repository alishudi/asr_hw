import torch_audiomentations
from torch import Tensor

from hw_asr.utils import ROOT_PATH
import gzip
import os, shutil
from speechbrain.utils.data_utils import download_file

from hw_asr.augmentations.base import AugmentationBase

URL_LINKS = {
    "musan": "https://openslr.elda.org/resources/17/musan.tar.gz",
    "rirs": "https://openslr.elda.org/resources/28/rirs_noises.zip"
}

#didnt test this augmentation properly, it might not work
class BackgroundNoise(AugmentationBase):
    def __init__(self, noise='rirs', *args, **kwargs):
        self.path = self.prepare_noise(noise)
        self._aug = torch_audiomentations.AddBackgroundNoise(background_paths=self.path, *args, **kwargs)

    def __call__(self, data: Tensor):
        try:
            x = data.unsqueeze(1).numpy()
            return Tensor(self._aug(x, sample_rate=16000)).squeeze(1)
        except:
            x = data.numpy()
            return Tensor(self._aug(x, sample_rate=16000))

    def prepare_noise(self, noise):
        assert noise in URL_LINKS

        data_dir = ROOT_PATH / "data" / noise
        data_dir.mkdir(exist_ok=True, parents=True)
        if noise == 'rirs':
            gzip_path = data_dir / 'rirs_noises.zip'
        else:
            gzip_path = data_dir / 'musan.tar.gz'
        if not os.path.exists(gzip_path):
            print('Downloading the noise data.')
            noise_url = URL_LINKS[noise]
            download_file(noise_url, gzip_path)
            print('Downloaded the noise data.')
        
        if noise == 'rirs':
            unzipped_path = data_dir / 'rirs_noises'
        else:
            unzipped_path = data_dir / 'musan.tar'
        if not os.path.exists(unzipped_path):
            with gzip.open(gzip_path, 'rb') as f_zipped:
                with open(unzipped_path, 'wb') as f_unzipped:
                    shutil.copyfileobj(f_zipped, f_unzipped)
            print('Unzipped the noise data.')

        return unzipped_path