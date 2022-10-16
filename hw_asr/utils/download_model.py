from hw_asr.utils import ROOT_PATH
import os, shutil
from speechbrain.utils.data_utils import download_file

URL = "https://drive.google.com/file/d/19-HH878SpTFWugKdMeV4cf5f4PTyFrG3/"
path = ROOT_PATH / 'default_test_model'
path.mkdir(exist_ok=True, parents=True)
download_file(URL, path / 'checkpoint.pth')