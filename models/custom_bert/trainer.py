import torch
import time
import os
from datasets import load_dataset
from torch.utils import tensorboard

from preprocessor import Preprocessor

from models.custom_bert.custom_bert import CustomBERT
from models.custom_bert.torch_bert import TorchBERT


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: str,
                 sentence_piece_model_path: str,
                 source_data_path: str,
                 batch_size: int,
                 max_input_size: int,
                 vocab_size: int,
                 n_warmup_steps: int,
                 n_training_steps: int,
                 learning_rate: float,
                 weight_decay: float,
                 clipping_value: float or None
                 ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.n_warmup_steps = n_warmup_steps
        self.n_training_steps = n_training_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clipping_value = clipping_value

        # print("Compiling model for faster training using Torch 2.0...")           # Not yet support for Windows, but on Linux could really speed up training.
        # self.compiled_model = torch.compile(model)
        # print("Model compiled.")
        self.preprocessor = Preprocessor(sentence_piece_model_path=sentence_piece_model_path,
                                         max_input_size=max_input_size,
                                         vocab_size=vocab_size)
        dataset = load_dataset("text", data_files={"train": source_data_path})
        dataset.set_format(type="torch", columns=["text"])

        self.dataloader = torch.utils.data.DataLoader(dataset["train"])
        self.train_subset_dataset, self.test_subset_dataset = torch.utils.data.random_split(
            dataset=self.dataloader.dataset,
            lengths=[len(self.dataloader.dataset) - 1000, 1000],
            generator=torch.Generator().manual_seed(0)
        )

        self.train_dataloader = torch.utils.data.DataLoader(self.train_subset_dataset, batch_size=self.batch_size)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_subset_dataset, batch_size=self.batch_size)

    def train(self, model_save_name: str, actual_step_start: int = 0):
        train_dataloader = iter(self.train_dataloader)

        criterion = torch.nn.NLLLoss()
        # using AdamW instead of Adam because its weight decay implementation is closer to what was used in BERT paper. Epsilon value is from RoBERTa paper
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=1e-6)
        # scaler = torch.cuda.amp.GradScaler()
        scheduler = self.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps,
        )

        tensorboard_writer = tensorboard.SummaryWriter(filename_suffix=model_save_name)

        # Main training loop
        for step in range(self.n_training_steps):
            input_sentences = next(train_dataloader)["text"]

            input_ids, input_ids_with_mask, input_mask = self.preprocessor.preprocess(input_sentences)
            input_ids, input_ids_with_mask, input_mask = input_ids.to(self.device), input_ids_with_mask.to(
                self.device), input_mask.to(self.device)

            # We set the label of the non-masked tokens as -100 to not take them into account when computing the loss. Please see torch.nn.NLLLoss documentation.
            labels = input_ids.clone()
            labels[input_mask == 0] = -100

            optimizer.zero_grad()

            # with torch.autocast(device_type='cuda', dtype=torch.float16):

            # Forward pass
            softmax_output, _ = self.model(input_ids_with_mask)
            # Computing the loss
            train_loss = criterion(softmax_output.transpose(1, 2), labels.long())

            train_accuracy_all_text_tokens, train_accuracy_masked_tokens = self.compute_metrics(
                softmax_output=softmax_output,
                input_ids=input_ids,
                input_mask=input_mask
            )

            if step % 50 == 0:
                tensorboard_writer.add_scalar(tag="Accuracy/train/masked_tokens",
                                              scalar_value=train_accuracy_masked_tokens,
                                              global_step=actual_step_start + step)
                tensorboard_writer.add_scalar(tag="Accuracy/train/all_text_tokens",
                                              scalar_value=train_accuracy_all_text_tokens,
                                              global_step=actual_step_start + step)
                tensorboard_writer.add_scalar(tag="Loss/train",
                                              scalar_value=train_loss,
                                              global_step=actual_step_start + step)
                tensorboard_writer.add_scalar(tag="Learning Rate",
                                              scalar_value=scheduler.get_last_lr()[0],
                                              global_step=actual_step_start + step)

            # scaler.scale(train_loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            train_loss.backward()
            if self.clipping_value:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipping_value)
            optimizer.step()

            scheduler.step()

            if step % 1000 == 0:
                test_dataloader = iter(self.test_dataloader)
                test_accuracy_all_text_tokens_list = []
                test_accuracy_masked_tokens_list = []
                test_loss_list = []
                for batch in test_dataloader:
                    with torch.no_grad():
                        input_sentences = batch["text"]
                        if len(input_sentences) != self.batch_size:
                            break
                        input_ids, input_ids_with_mask, input_mask = self.preprocessor.preprocess(input_sentences)
                        input_ids, input_ids_with_mask, input_mask = input_ids.to(self.device), input_ids_with_mask.to(
                            self.device), input_mask.to(self.device)

                        softmax_output, _ = self.model(input_ids_with_mask)

                        # We set the label of the non-masked tokens as -100 to not take them into account when computing the loss. For more details check torch.nn.NLLLoss documentation.
                        labels = input_ids.clone()
                        labels[input_mask == 0] = -100

                        # Computing the loss
                        test_loss = criterion(softmax_output.transpose(1, 2), labels.long())

                        test_accuracy_all_text_tokens, test_accuracy_masked_tokens = self.compute_metrics(
                            softmax_output=softmax_output,
                            input_ids=input_ids,
                            input_mask=input_mask
                        )

                        test_accuracy_all_text_tokens_list.append(test_accuracy_all_text_tokens)
                        test_accuracy_masked_tokens_list.append(test_accuracy_masked_tokens)
                        test_loss_list.append(test_loss)

                average_test_loss = torch.mean(torch.Tensor(test_loss_list))
                average_test_accuracy_masked_tokens = torch.mean(torch.Tensor(test_accuracy_masked_tokens_list))
                average_test_accuracy_all_text_tokens = torch.mean(torch.Tensor(test_accuracy_all_text_tokens_list))

                print(f" Learning Rate= {scheduler.get_last_lr()[0]:1.10f}",
                      f" Step {step: <10} "
                      f" loss= {average_test_loss:1.5f},"
                      f" masked_tokens_accuracy= {average_test_accuracy_masked_tokens:1.5f},"
                      f" all_text_tokens_accuracy= {average_test_accuracy_all_text_tokens:1.5f}")

                tensorboard_writer.add_scalar(tag="Accuracy/test/masked_tokens",
                                              scalar_value=average_test_accuracy_masked_tokens,
                                              global_step=actual_step_start + step)
                tensorboard_writer.add_scalar(tag="Accuracy/test/all_text_tokens",
                                              scalar_value=average_test_accuracy_all_text_tokens,
                                              global_step=actual_step_start + step)
                tensorboard_writer.add_scalar(tag="Loss/test",
                                              scalar_value=average_test_loss,
                                              global_step=actual_step_start + step)

            if step % 20000 == 0:
                self.save_model(model_save_name=model_save_name, step=actual_step_start + step)

        tensorboard_writer.close()

    def compute_metrics(self, softmax_output, input_ids, input_mask):
        predictions = torch.argmax(softmax_output, dim=-1)
        padding_token_id = self.preprocessor.SPECIAL_TOKENS_IDS["<pad>"]
        is_correct_prediction = predictions == input_ids
        accuracy_all_text_tokens = torch.sum(is_correct_prediction[input_ids != padding_token_id]) / torch.numel(
            is_correct_prediction[input_ids != padding_token_id])
        accuracy_masked_tokens = torch.sum(is_correct_prediction * input_mask) / torch.sum(input_mask)

        return accuracy_all_text_tokens, accuracy_masked_tokens

    def save_model(self, model_save_name: str, step: int):
        if not os.path.exists("./trained_models"):
            os.mkdir("./trained_models")
        if not os.path.exists(f"./trained_models/{model_save_name}"):
            os.mkdir(f"./trained_models/{model_save_name}")

        torch.save(self.model, f"./trained_models/{model_save_name}/{step}.pt")

    @staticmethod
    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        """
        Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
        a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

        => Found here https://huggingface.co/transformers/v3.5.1/_modules/transformers/optimization.html

        Args:
            optimizer (:class:`~torch.optim.Optimizer`):
                The optimizer for which to schedule the learning rate.
            num_warmup_steps (:obj:`int`):
                The number of steps for the warmup phase.
            num_training_steps (:obj:`int`):
                The total number of training steps.
            last_epoch (:obj:`int`, `optional`, defaults to -1):
                The index of the last epoch when resuming training.

        Return:
            :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
        """

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def main():
    sentence_piece_model_path = "../../tokenizer/english_spm_lowercase.model"
    source_data_path = "training_data/data/base_data_lowercase.txt"

    # vocab_size = 30000
    # n_encoder_layers = 8
    # d_model = 432
    # d_ff_hidden = d_model * 2
    # h = 12
    # max_input_size = 128

    vocab_size = 30000
    n_encoder_layers = 12
    d_model = 768
    d_ff_hidden = d_model * 4
    h = 12
    max_input_size = 100

    device = "cuda:0"
    batch_size = 35
    n_warmup_steps = 50000
    n_training_steps = 3000000
    weight_decay = 0.05
    learning_rate = 4e-6  #  4e-5
    clipping_value = 1

    model_type = "torch"        # custom or pytorch
    # model_name = f"{model_type}_BERT_w_gradient_clipping_{time.time()}"
    model_name = f"{model_type}_BERT_{time.time()}"

    actual_step_start = 0  # 1720000
    load_checkpoint = None   # f"trained_models/{model_type}_BERT_w_gradient_clipping_1679888363.7064824_GOOD/1720000.pt"

    if model_type == "custom":
        model = CustomBERT(device=device,
                           vocab_size=vocab_size,
                           n_encoder_layers=n_encoder_layers,
                           d_model=d_model,
                           d_ff_hidden=d_ff_hidden,
                           h=h,
                           max_input_size=max_input_size)

    elif model_type == "torch":
        model = TorchBERT(device=device,
                          vocab_size=vocab_size,
                          n_encoder_layers=n_encoder_layers,
                          d_model=d_model,
                          d_ff_hidden=d_ff_hidden,
                          h=h,
                          max_input_size=max_input_size)
    else:
        raise NotImplementedError()

    if load_checkpoint:
        model.load_state_dict(torch.load(load_checkpoint).state_dict())

    trainer = Trainer(model=model,
                      device=device,
                      sentence_piece_model_path=sentence_piece_model_path,
                      source_data_path=source_data_path,
                      batch_size=batch_size,
                      max_input_size=max_input_size,
                      vocab_size=vocab_size,
                      n_warmup_steps=n_warmup_steps,
                      n_training_steps=n_training_steps,
                      weight_decay=weight_decay,
                      learning_rate=learning_rate,
                      clipping_value=clipping_value)

    trainer.train(model_save_name=model_name,
                  actual_step_start=actual_step_start)


main()
