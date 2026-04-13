"""
infer.py

Load a trained Vanilla MTL model and run inference on custom text inputs.

Usage examples:
    # SST-2 sentiment
    python infer.py --task sst2 --texts "This movie was absolutely fantastic!" "I hated every minute."

    # CoLA grammar
    python infer.py --task cola --texts "The cat sat on the mat." "Cat the sat mat on."

    # QQP paraphrase (comma-separated pairs)
    python infer.py --task qqp --pairs "What is AI?,What does AI stand for?" "How old is earth?,What is the age of the universe?"

    # MNLI entailment (premise:::hypothesis)
    python infer.py --task mnli --pairs "A dog is chasing a ball.:::The dog is playing." "It's raining.:::The sun is shining."
"""

import argparse
import torch
from pathlib import Path
from transformers import DistilBertTokenizerFast

from model import VanillaMTLModel, TASK_NUM_LABELS

# ── Label maps ────────────────────────────────────────────────────────────────

LABEL_MAPS = {
    "sst2": {0: "NEGATIVE", 1: "POSITIVE"},
    "cola": {0: "UNACCEPTABLE", 1: "ACCEPTABLE"},
    "qqp":  {0: "NOT_DUPLICATE", 1: "DUPLICATE"},
    "mnli": {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"},
}


def load_model(checkpoint_path: str, model_name: str = "distilbert-base-uncased"):
    model = VanillaMTLModel(model_name=model_name)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, tokenizer, texts, task, device, max_length=128, batch_size=32):
    from train_vanilla_mtl import predict  # reuse predict function
    preds, probs = predict(model, texts, task, tokenizer, device, batch_size, max_length)
    return preds, probs


def main():
    parser = argparse.ArgumentParser(description="Vanilla MTL Inference")
    parser.add_argument("--checkpoint", default="./outputs/vanilla_mtl/best_model.pt")
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--task", required=True, choices=list(TASK_NUM_LABELS.keys()))
    parser.add_argument("--texts", nargs="+", default=None,
                        help="Single sentences (for sst2 / cola)")
    parser.add_argument("--pairs", nargs="+", default=None,
                        help="Comma-separated pairs for qqp; triple-colon-separated for mnli")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name)
    model = load_model(args.checkpoint, args.model_name)
    model.to(device)

    label_map = LABEL_MAPS[args.task]

    # Build input list
    if args.task in ("sst2", "cola"):
        if not args.texts:
            parser.error("--texts required for sst2/cola tasks")
        inputs = args.texts
    else:
        if not args.pairs:
            parser.error("--pairs required for qqp/mnli tasks")
        sep = ":::" if args.task == "mnli" else ","
        inputs = [tuple(p.split(sep, 1)) for p in args.pairs]

    preds, probs = run_inference(
        model, tokenizer, inputs, args.task, device, args.max_length, args.batch_size
    )

    print(f"\nResults for task: {args.task.upper()}")
    print("-" * 60)
    for i, (inp, pred, prob) in enumerate(zip(inputs, preds, probs)):
        if isinstance(inp, tuple):
            print(f"[{i+1}] Input A: {inp[0]}")
            print(f"     Input B: {inp[1]}")
        else:
            print(f"[{i+1}] Input: {inp}")
        print(f"     Prediction: {label_map[pred]}")
        prob_str = "  ".join(f"{label_map[j]}: {p:.3f}" for j, p in enumerate(prob))
        print(f"     Probabilities: {prob_str}")
        print()


if __name__ == "__main__":
    main()
