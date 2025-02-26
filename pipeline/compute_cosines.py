import argparse
import wandb
import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # TODO: load the test data, harmless, harmful and jailbreak
    # TODO: load the direction vectors
    # TODO: compute cosine distances
    # TODO: store the results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument(
        "--model_name", type=str, required=True, help="Path to the model"
    )
    args = parser.parse_args()

    wandb.init(
        project="safety_refusal_cosine",
        name=" ".join([args.model_name]),
        config=args,
    )

    main(args)
