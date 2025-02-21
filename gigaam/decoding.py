from typing import List, Optional, Tuple, Union

import torch
from ctcdecode import CTCBeamDecoder
from sentencepiece import SentencePieceProcessor
from torch import Tensor

from .decoder import CTCHead, RNNTHead

MS_PER_TS = 40
MS_DELAY = 0

class Tokenizer:
    """
    Tokenizer for converting between text and token IDs.
    The tokenizer can operate either character-wise or using a pre-trained SentencePiece model.
    """

    def __init__(self, vocab: List[str], model_path: Optional[str] = None):
        self.charwise = model_path is None
        if self.charwise:
            self.vocab = vocab
        else:
            self.model = SentencePieceProcessor()
            self.model.load(model_path)

    def decode(self, tokens: List[int]) -> str:
        """
        Convert a list of token IDs back to a string.
        """
        if self.charwise:
            return "".join(self.vocab[tok] for tok in tokens)
        return self.model.decode(tokens)

    def __len__(self):
        """
        Get the total number of tokens in the vocabulary.
        """
        return len(self.vocab) if self.charwise else len(self.model)


class CTCGreedyDecoding:
    """
    Class for performing greedy decoding of CTC outputs.
    """

    def __init__(self, vocabulary: List[str], model_path: Optional[str] = None):
        self.tokenizer = Tokenizer(vocabulary, model_path)
        self.blank_id = len(self.tokenizer)

    @torch.inference_mode()
    def decode(self, head: CTCHead, encoded: Tensor, lengths: Tensor, return_ts: bool = False) -> List[str]:
        """
        Decode the output of a CTC model into a list of hypotheses.
        """
        if return_ts:
            raise NotImplementedError("CTC greedy decoding with timestamps is not implemented")
        
        log_probs = head(encoder_output=encoded)
        assert (
            len(log_probs.shape) == 3
        ), f"Expected log_probs shape {log_probs.shape} == [B, T, C]"
        b, _, c = log_probs.shape
        assert (
            c == len(self.tokenizer) + 1
        ), f"Num classes {c} != len(vocab) + 1 {len(self.tokenizer) + 1}"
        labels = log_probs.argmax(dim=-1, keepdim=False)

        skip_mask = labels != self.blank_id
        skip_mask[:, 1:] = torch.logical_and(
            skip_mask[:, 1:], labels[:, 1:] != labels[:, :-1]
        )
        for length in lengths:
            skip_mask[length:] = 0

        pred_texts: List[str] = []
        for i in range(b):
            pred_texts.append(
                "".join(self.tokenizer.decode(labels[i][skip_mask[i]].cpu().tolist()))
            )
        return pred_texts


class CTCBeamLMDecoding:
    """
    Class for performing beamsearch decoding ngram LM rescoring of CTC outputs.
    """

    def __init__(
            self,
            vocabulary: List[str],
            model_path: Optional[str] = None,
            ngram_arpa_path: Optional[str] = None,
            alpha: float = 0.,
            beta: float = 0.,
            cutoff_top_n: int = 40,
            cutoff_prob: float = 1.0,
            beam_width: int = 100,
            num_processes: int = 4,
        ):
        self.tokenizer = Tokenizer(vocabulary, model_path)
        self.blank_id = len(self.tokenizer)
        self.decoder = CTCBeamDecoder(
                labels="".join(vocabulary) + "_",
                model_path=ngram_arpa_path,
                alpha=alpha,
                beta=beta,
                cutoff_top_n=cutoff_top_n,
                cutoff_prob=cutoff_prob,
                beam_width=beam_width,
                num_processes=num_processes,
                blank_id=self.blank_id,
                log_probs_input=True
        )

    def __ts_to_ms(self, ts: float) -> float:
        return ts * MS_PER_TS + MS_DELAY

    def get_word_timestamps(self, text: str, char_timestamps: List[float]) -> List[dict]:
        words = text.split(" ")
        word_timestamps = []
        char_index = 0

        for word in words:
            if word != "":
                start_ts = self.__ts_to_ms(char_timestamps[char_index])
                end_ts = self.__ts_to_ms(char_timestamps[char_index + len(word) - 1]) + 1

                word_timestamps.append({
                    "word": word,
                    "start_ms": start_ts,
                    "end_ms": end_ts
                })

            char_index += len(word) + 1

        return word_timestamps

    @torch.inference_mode()
    def decode(self, head: CTCHead, encoded: Tensor, lengths: Tensor, return_ts: bool = False) -> Union[List[str], List[List[dict]]]:
        """
        Decode the output of a CTC model into a list of hypotheses.
        """
        log_probs = head(encoder_output=encoded)
        assert (
            len(log_probs.shape) == 3
        ), f"Expected log_probs shape {log_probs.shape} == [B, T, C]"
        b, _, c = log_probs.shape
        assert (
            c == len(self.tokenizer) + 1
        ), f"Num classes {c} != len(vocab) + 1 {len(self.tokenizer) + 1}"

        beam_results, beam_scores, timestamps, out_lens = self.decoder.decode(log_probs)

        pred_texts: List[str] = []
        word_timestamps: List[List[dict]] = []
        for i in range(b):
            text = "".join(self.tokenizer.decode(beam_results[i][0][:out_lens[i][0]].cpu().tolist()))
            pred_texts.append(text)
            if return_ts:
                word_timestamps.append(self.get_word_timestamps(text, timestamps[i][0][:out_lens[i][0]].cpu().tolist()))
        if return_ts:
            return word_timestamps
        return pred_texts


class RNNTGreedyDecoding:
    def __init__(
        self,
        vocabulary: List[str],
        model_path: Optional[str] = None,
        max_symbols_per_step: int = 10,
    ):
        """
        Class for performing greedy decoding of RNN-T outputs.
        """
        self.tokenizer = Tokenizer(vocabulary, model_path)
        self.blank_id = len(self.tokenizer)
        self.max_symbols = max_symbols_per_step

    def _greedy_decode(self, head: RNNTHead, x: Tensor, seqlen: Tensor) -> str:
        """
        Internal helper function for performing greedy decoding on a single sequence.
        """
        hyp: List[int] = []
        dec_state: Optional[Tensor] = None
        last_label: Optional[Tensor] = None
        for t in range(seqlen):
            f = x[t, :, :].unsqueeze(1)
            not_blank = True
            new_symbols = 0
            while not_blank and new_symbols < self.max_symbols:
                g, hidden = head.decoder.predict(last_label, dec_state)
                k = head.joint.joint(f, g)[0, 0, 0, :].argmax(0).item()
                if k == self.blank_id:
                    not_blank = False
                else:
                    hyp.append(k)
                    dec_state = hidden
                    last_label = torch.tensor([[hyp[-1]]]).to(x.device)
                    new_symbols += 1

        return self.tokenizer.decode(hyp)

    @torch.inference_mode()
    def decode(self, head: RNNTHead, encoded: Tensor, enc_len: Tensor, return_ts: bool = False) -> List[str]:
        """
        Decode the output of an RNN-T model into a list of hypotheses.
        """
        if return_ts:
            raise NotImplementedError("RNN-T decoding with timestamps is not implemented")
        b = encoded.shape[0]
        pred_texts = []
        encoded = encoded.transpose(1, 2)
        for i in range(b):
            inseq = encoded[i, :, :].unsqueeze(1)
            pred_texts.append(self._greedy_decode(head, inseq, enc_len[i]))
        return pred_texts
