from typing import List, NamedTuple

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

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
        hypos: List[Hypothesis] = []
        # mostly copied from the seminar

        paths = {('', self.EMPTY_TOK): 1}
        for i in range(probs_length):
            #extend and merge
            new_paths = {}
            for next_char_ind, next_char_prob in enumerate(probs[i]):
                next_char = self.ind2char[next_char_ind]
                for (text, last_char), path_prob in paths.items():
                    new_prefix = text if next_char == last_char else (text + last_char)
                    new_prefix = new_prefix.replace(self.EMPTY_TOK, '')
                    new_paths[(new_prefix, next_char)] += path_prob * next_char_prob
            paths = new_paths

            #truncate beam
            paths = dict(sorted(paths.items(), key=lambda x: x[1])[-beam_size:])

        return [Hypothesis(prefix, score) for (prefix, _), score in sorted(paths.items(), key=lambda x: -x[1])]
        return sorted(hypos, key=lambda x: x.prob, reverse=True)
