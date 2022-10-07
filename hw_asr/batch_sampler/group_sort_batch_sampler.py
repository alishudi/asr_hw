from torch.utils.data import Sampler


class GroupLengthBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, batches_per_group=20):
        super().__init__(data_source)
        # TODO: your code here (optional)
        for sample in data_source:
            sample['audio'].shape[0]
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
