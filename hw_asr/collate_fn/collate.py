import logging
from typing import List
from torch.nn.utils.rnn import pad_sequence
import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """
    
    result_batch = {}

    for key in dataset_items[0]:
        if key not in ['spectrogram', 'text_encoded']:
            result_batch[key] = [d[key] for d in dataset_items]
        else:
            result_batch[key] = [torch.squeeze(d[key], dim=0).T for d in dataset_items]
            result_batch[key + '_length'] = torch.tensor([d[key].shape[-1] for d in dataset_items])

    result_batch['spectrogram'] = pad_sequence(result_batch['spectrogram'], batch_first=True).transpose(1, 2)
    result_batch['text_encoded'] = pad_sequence(result_batch['text_encoded'], batch_first=True)

    return result_batch