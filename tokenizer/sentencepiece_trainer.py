import sentencepiece as spm
import time


class SentencePieceTokenizerTrainer:
    SPECIAL_TOKENS_IDS = {
        "<pad>": 0,
        "<unk>": 1,
        "<s>": 2,
        "</s>": 3,
        "[MASK]": 4,
        "[CLS]": 5,
        "[SEP]": 6
    }

    def train_model(self, input_data_path: str, vocab_size: int, n_sentences_to_use: int, model_prefix: str):
        spm.SentencePieceTrainer.train(input=input_data_path,
                                       model_prefix=model_prefix,
                                       input_sentence_size=n_sentences_to_use,
                                       vocab_size=vocab_size,
                                       pad_id=self.SPECIAL_TOKENS_IDS["<pad>"],
                                       unk_id=self.SPECIAL_TOKENS_IDS["<unk>"],
                                       bos_id=self.SPECIAL_TOKENS_IDS["<s>"],
                                       eos_id=self.SPECIAL_TOKENS_IDS["</s>"],
                                       control_symbols="[MASK],[CLS],[SEP]"
                                       )

    @staticmethod
    def lower_case_data(input_data_path: str, output_data_path: str):
        with open(input_data_path, "r", encoding="utf-8") as source_file:
            with open(output_data_path, "w", encoding="utf-8") as target_file:
                for x in source_file.readlines():
                    target_file.write(x.lower())


def main():
    spm_tokenizer = SentencePieceTokenizerTrainer()
    spm_tokenizer.lower_case_data(input_data_path="../training_data/data/base_data.txt",
                                  output_data_path="../training_data/data/base_data_lowercase.txt")
    spm_tokenizer.train_model(input_data_path="../training_data/data/base_data_lowercase.txt",
                              model_prefix=f"{time.time()}_english_spm_lowercase",
                              vocab_size=30000,
                              n_sentences_to_use=5000000)


main()
