from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel

#took architecture from here https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/deepspeech2.html
#T, F -> F, T
class DeepSpeech2(BaseModel):
    def __init__(self, n_feats, n_class, **batch):
        super().__init__(n_feats, n_class, **batch)

        self.conv = Sequential(
            nn.Conv2d(1, 32, (41, 11), (2, 2), padding="same"),
            nn.BatchNorm2d(32, eps=1e-3, momentum=0.99),
            nn.ReLU(),
            nn.Conv2d(32, 32, (21, 11), (2, 1), padding="same"),
            nn.BatchNorm2d(32, eps=1e-3, momentum=0.99),
            nn.ReLU()
            )
        self.gru = nn.GRU(
            num_layers=5,
            hidden_size=800,
            bidirectional=True,
            dropout=0.5,
            batch_first=True
            )
        self.head = Sequential(
            nn.Linear(in_features=1600, out_features=1600),
            nn.Linear(in_features=1600, out_features=n_class)
        )

    def forward(self, spectrogram, **batch):
        x = spectrogram
        print(x.shape)
        x = self.conv(x)
        print(x.shape)
        x, _ = self.gru(x)
        print(x.shape)
        x = self.head(x)

        return {"logits": x}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
