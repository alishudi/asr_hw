import gzip
import os, shutil
from hw_asr.utils import ROOT_PATH
from speechbrain.utils.data_utils import download_file

#slightly rewritten code from here https://github.com/kensho-technologies/pyctcdecode/blob/main/tutorials/03_eval_performance.ipynb
def load_lm():
    data_dir = ROOT_PATH / "data" / "lm"
    data_dir.mkdir(exist_ok=True, parents=True)
    lm_gzip_path = data_dir / '3-gram.pruned.1e-7.arpa.gz'
    if not os.path.exists(lm_gzip_path):
        print('Downloading pruned 3-gram model.')
        lm_url = 'http://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz'
        download_file(lm_url, lm_gzip_path)
        print('Downloaded the 3-gram language model.')
    else:
        print('Pruned .arpa.gz already exists.')

    # NOTE: since out nemo vocabulary is all lowercased, we need to convert all librispeech data as well
    uppercase_lm_path = data_dir / '3-gram.pruned.1e-7.arpa'
    if not os.path.exists(uppercase_lm_path):
        with gzip.open(lm_gzip_path, 'rb') as f_zipped:
            with open(uppercase_lm_path, 'wb') as f_unzipped:
                shutil.copyfileobj(f_zipped, f_unzipped)
        print('Unzipped the 3-gram language model.')
    else:
        print('Unzipped .arpa already exists.')

    lm_path = data_dir / 'lowercase_3-gram.pruned.1e-7.arpa'
    if not os.path.exists(lm_path):
        with open(uppercase_lm_path, 'r') as f_upper:
            with open(lm_path, 'w') as f_lower:
                for line in f_upper:
                    f_lower.write(line.lower())
    print('Converted language model file to lowercase.')

    # download unigram vocab
    unigram_path = data_dir / 'librispeech-vocab.txt'
    if not os.path.exists(unigram_path):
        print('Downloading unigram vocab.')
        uni_url = 'http://www.openslr.org/resources/11/librispeech-vocab.txt'
        download_file(uni_url, unigram_path)
        print('Downloaded unigram vocab.')
    else:
        print('Unigram vocab already exists.')

    return lm_path, unigram_path