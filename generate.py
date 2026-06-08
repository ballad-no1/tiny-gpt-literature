import argparse
from pathlib import Path
import torch

from model import TinyGPT


def main():
    parser = argparse.ArgumentParser(description="Generate text with trained Tiny GPT")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/tiny_gpt.pt")
    parser.add_argument("--prompt", type=str, default="ROMEO:")
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError("Checkpoint not found. Run `python train.py` first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint["config"]
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    # torch may save JSON-like dict keys as strings in some environments
    itos = {int(k): v for k, v in itos.items()}

    model = TinyGPT(**config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    encoded = [stoi[ch] for ch in args.prompt if ch in stoi]
    if len(encoded) == 0:
        encoded = [0]
    idx = torch.tensor([encoded], dtype=torch.long, device=device)
    out_idx = model.generate(idx, args.max_new_tokens, temperature=args.temperature)[0].tolist()
    text = "".join(itos[i] for i in out_idx)
    print(text)


if __name__ == "__main__":
    main()
