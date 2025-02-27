import argparse
import wandb
import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import random
from dataclasses import dataclass
from dataset.load_dataset import load_dataset_split
from pipeline.submodules.select_direction import get_refusal_scores
from pipeline.model_utils.qwen_model import (
    tokenize_instructions_qwen_chat,
    QWEN_REFUSAL_TOKS,
)
from functools import partial
from pipeline.submodules.generate_directions import get_mean_activations


def tokenize_instructions(tokenizer, instructions):
    return tokenizer(
        instructions,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )


class ModelAndTokenizer:
    model_name: str
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer

    def __init__(self, model_name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = self.model.eval()
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token is None:
            if "qwen" in model_name.lower():
                self.tokenizer.pad_token = "<|extra_0|>"
                self.tokenizer.pad_token_id = self.tokenizer.eod_id
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_instructions_fn(self):
        model_to_tokenize_fn = {"Qwen/Qwen-1_8B-Chat": tokenize_instructions_qwen_chat}
        if self.model_name not in model_to_tokenize_fn:
            return partial(tokenize_instructions, self.tokenizer)
        return partial(model_to_tokenize_fn[self.model_name], tokenizer=self.tokenizer)

    def refusal_toks(self):
        model_to_tokens = {"Qwen/Qwen-1_8B-Chat": QWEN_REFUSAL_TOKS}
        return model_to_tokens[self.model_name]

    def get_module(self, module):
        if "qwen" in self.model_name.lower():
            model_to_modules = {"layer": self.model.transformer.h}
        return model_to_modules[module]


def get_average_reprs(model_and_tokenizer, dataset):
    return get_mean_activations(
        model_and_tokenizer.model,
        model_and_tokenizer.tokenizer,
        dataset,
        model_and_tokenizer.tokenize_instructions_fn(),
        model_and_tokenizer.get_module("layer"),
        positions=[-1],  # future: select best token
    )


def load_and_sample_datasets(n_train, n_val):
    """
    Load datasets and sample them based on the configuration.

    Returns:
        Tuple of datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """
    # future: change to numpy rng
    random.seed(42)
    harmful_train = random.sample(
        load_dataset_split(harmtype="harmful", split="train", instructions_only=True),
        n_train,
    )
    harmless_train = random.sample(
        load_dataset_split(harmtype="harmless", split="train", instructions_only=True),
        n_train,
    )
    harmful_val = random.sample(
        load_dataset_split(harmtype="harmful", split="val", instructions_only=True),
        n_val,
    )
    harmless_val = random.sample(
        load_dataset_split(harmtype="harmless", split="val", instructions_only=True),
        n_val,
    )
    return harmful_train, harmless_train, harmful_val, harmless_val


def filter_data(model_base, harmful_train, harmless_train, harmful_val, harmless_val):
    """
    Filter datasets based on refusal scores.

    Returns:
        Filtered datasets: (harmful_train, harmless_train, harmful_val, harmless_val)
    """

    def filter_examples(dataset, scores, threshold, comparison):
        return [
            inst
            for inst, score in zip(dataset, scores.tolist())
            if comparison(score, threshold)
        ]

    harmful_train_scores = get_refusal_scores(
        model_base.model,
        harmful_train,
        model_base.tokenize_instructions_fn(),
        model_base.refusal_toks(),
    )
    harmless_train_scores = get_refusal_scores(
        model_base.model,
        harmless_train,
        model_base.tokenize_instructions_fn(),
        model_base.refusal_toks(),
    )
    harmful_train = filter_examples(
        harmful_train, harmful_train_scores, 0, lambda x, y: x > y
    )
    harmless_train = filter_examples(
        harmless_train, harmless_train_scores, 0, lambda x, y: x < y
    )
    """
    harmful_val_scores = get_refusal_scores(
        model_base.model,
        harmful_val,
        model_base.tokenize_instructions_fn(),
        model_base.refusal_toks(),
    )
    harmless_val_scores = get_refusal_scores(
        model_base.model,
        harmless_val,
        model_base.tokenize_instructions_fn(),
        model_base.refusal_toks(),
    )
    harmful_val = filter_examples(
        harmful_val, harmful_val_scores, 0, lambda x, y: x > y
    )
    harmless_val = filter_examples(
        harmless_val, harmless_val_scores, 0, lambda x, y: x < y
    )
    """
    return harmful_train, harmless_train  # , harmful_val, harmless_val


def main(args):
    exp_dir = "_".join(
        [
            args.base_model_name.replace("/", "__"),
            args.chat_model_name.replace("/", "__"),
        ]
    )
    output_folder = os.path.join(args.output_folder, exp_dir)
    os.makedirs(output_folder, exist_ok=True)
    wandb.run.summary["final_output_folder"] = output_folder

    base_model = ModelAndTokenizer(args.base_model_name)
    chat_model = ModelAndTokenizer(args.chat_model_name)

    harmful_train, harmless_train, harmful_val, harmless_val = load_and_sample_datasets(
        args.n_train, args.n_val
    )
    print("Datasets loaded...")
    # Filter datasets based on refusal scores
    (
        harmful_refused_train,
        harmless_non_refused_train,
        # harmful_refused_val,
        # harmless_non_refused_val,
    ) = filter_data(
        chat_model, harmful_train, harmless_train, harmful_val, harmless_val
    )
    print("Datasets filtered...")
    h_base = get_average_reprs(base_model, harmful_train) - get_average_reprs(
        base_model, harmless_train
    )
    torch.save(h_base, os.path.join(output_folder, "h_base.pt"))
    r_chat_base = get_average_reprs(
        chat_model, harmful_refused_train
    ) - get_average_reprs(base_model, harmful_refused_train)
    d_ift = get_average_reprs(
        chat_model, harmful_train + harmless_train
    ) - get_average_reprs(base_model, harmful_train + harmless_train)
    # r = x_refused - x_base - d_ift
    r_chat_base_ift = r_chat_base - d_ift
    torch.save(r_chat_base, os.path.join(output_folder, "r_chat_base.pt"))
    torch.save(r_chat_base_ift, os.path.join(output_folder, "r_chat_base_ift.pt"))
    r_chat = get_average_reprs(chat_model, harmful_refused_train) - get_average_reprs(
        chat_model, harmless_non_refused_train
    )
    torch.save(r_chat, os.path.join(output_folder, "r_chat.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument(
        "--base_model_name", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--chat_model_name", type=str, required=True, help="Path to the model"
    )
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--n_train", type=int, default=128)
    parser.add_argument("--n_val", type=int, default=32)
    args = parser.parse_args()

    wandb.init(
        project="safety_refusal_directions",
        name=" ".join([args.base_model_name, args.chat_model_name]),
        config=args,
    )

    main(args)
