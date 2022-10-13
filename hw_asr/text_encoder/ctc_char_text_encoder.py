from typing import List, NamedTuple
from collections import defaultdict
from pyctcdecode import build_ctcdecoder
import kenlm

import torch

from .char_text_encoder import CharTextEncoder
from hw_asr.utils.download_lm import load_lm


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "" #pyctcdecoder doesnt recognize "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        #load language model
        lm_path, vocab_path = load_lm()
        # load unigram list
        with open(vocab_path) as f:
            unigram_list = [t.lower() for t in f.read().strip().split("\n")]
        self.bs_lm = build_ctcdecoder(
            vocab,
            str(lm_path),
            unigram_list,
        )

    def ctc_decode(self, inds: List[int]) -> str:
        decoded = []
        last_tok = self.EMPTY_TOK
        for tok in inds:
            if tok != last_tok:
                last_tok = tok
                if self.ind2char[tok] != self.EMPTY_TOK:
                    decoded.append(self.ind2char[tok])
        decoded = ''.join(decoded)
        return decoded

    def ctc_beam_search(self, probs: torch.tensor, probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        print(probs[:probs_length].shape, probs[:probs_length])
        beams = self.bs_lm.decode_beams(probs[:probs_length], beam_width=beam_size)
        hypos = [Hypothesis(text, combined_score) for text, _, _, _, combined_score in beams]
        return hypos
