from datasets import load_dataset
from datasets import concatenate_datasets
import time
import os.path


def load_train_data(output_folder: str, output_prefix: str):
    start_time = time.time()
    dataset_lm1b_train = load_dataset("lm1b")["train"]
    dataset_bookcorpus_train = load_dataset("bookcorpus")["train"]
    datasets = [dataset_lm1b_train, dataset_bookcorpus_train]

    final_dataset = concatenate_datasets(datasets)
    final_dataset.shuffle(seed=0)

    sentences = list(final_dataset["text"])

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    with open(f"{output_folder}/{output_prefix}", "w", encoding="utf-8") as f:
        for i, s in enumerate(sentences):
            f.write(s + "\n")

    print(f"Successfully loaded data ({len(sentences)} sentences) in {time.time() - start_time:1.2f} seconds.")


def main():
    load_train_data(output_folder=f"./data", output_prefix=f"{time.time()}_base_data.txt")


main()
