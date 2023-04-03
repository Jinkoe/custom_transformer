from itertools import repeat, chain, islice
import sentencepiece as spm
import torch

from tokenizer.sentencepiece_trainer import SentencePieceTokenizerTrainer


class Preprocessor:
    SPECIAL_TOKENS_IDS = SentencePieceTokenizerTrainer.SPECIAL_TOKENS_IDS

    def __init__(self, sentence_piece_model_path: str, max_input_size: int, vocab_size: int):
        self.max_input_size = max_input_size
        self.vocab_size = vocab_size
        self.tokenizer = spm.SentencePieceProcessor(model_file=sentence_piece_model_path)

        self.first_text_token_id = max([y for x, y in self.SPECIAL_TOKENS_IDS.items()]) + 1
        print(self.first_text_token_id)

        # Original BERT masking probabilities
        self.mask_token_as_mask_probability = 0.12
        self.mask_token_as_random_token_probability = 0.015
        self.mask_token_as_original_token_probability = 0.015

    def preprocess(self, sentences_batch: list):
        batch_size = len(sentences_batch)

        encoded_sentences_batch = [self.tokenizer.encode(s) for s in sentences_batch]
        encoded_sentences_batch = self.truncate_and_pad(encoded_sentences_batch)
        input_ids = torch.IntTensor(encoded_sentences_batch)

        input_ids_with_mask, input_mask = self.apply_mask(input_ids=input_ids, batch_size=batch_size)

        return input_ids, input_ids_with_mask, input_mask

    def truncate_and_pad(self, encoded_sentences_batch: list):
        return [list(islice(chain(x, repeat(self.SPECIAL_TOKENS_IDS["<pad>"])), self.max_input_size)) for x in
                encoded_sentences_batch]

    def apply_mask(self, input_ids: torch.IntTensor, batch_size: int):
        mask_probability = torch.rand((batch_size, self.max_input_size))

        # We don't want to mask padding tokens (<pad>)
        mask_probability[input_ids == self.SPECIAL_TOKENS_IDS["<pad>"]] = 1

        # Finding where to apply different types of masking strategies.
        # We don't need to find where to "mask as original token" since we won't change the original token anyway.
        where_mask_as_mask = mask_probability <= self.mask_token_as_mask_probability
        where_mask_as_random_token = torch.logical_and(
            mask_probability > self.mask_token_as_mask_probability,
            mask_probability <= self.mask_token_as_mask_probability + self.mask_token_as_random_token_probability)

        # Masking tokens with the [MASK] token id
        input_ids_with_mask = input_ids.clone()
        input_ids_with_mask[where_mask_as_mask] = self.SPECIAL_TOKENS_IDS["[MASK]"]

        # Masking tokens with random tokens ids (excluding special tokens)
        random_tokens = torch.randint(low=self.first_text_token_id,
                                      high=self.vocab_size - 1,
                                      size=(batch_size, self.max_input_size),
                                      dtype=torch.int32)

        input_ids_with_mask[where_mask_as_random_token] = random_tokens[where_mask_as_random_token]

        # Generating the input_mask tensor,
        # containing 1 for masked tokens (either masked as mask, masked as original token or masked as random token),
        # else 0
        input_mask = torch.zeros((batch_size, self.max_input_size))
        input_mask[mask_probability <= (self.mask_token_as_mask_probability
                   + self.mask_token_as_original_token_probability
                   + self.mask_token_as_random_token_probability)] = 1

        return input_ids_with_mask, input_mask
