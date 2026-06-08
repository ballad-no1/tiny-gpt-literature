import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

from model import TinyGPT


class NextTokenDataset(Dataset):
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + self.block_size + 1]
        return x, y


def build_vocab(text: str):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return chars, stoi, itos


@torch.no_grad()
def estimate_loss(model, loader, device, eval_iters: int):
    model.eval()
    losses = []
    for i, (xb, yb) in enumerate(loader):
        if i >= eval_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def main():
    parser = argparse.ArgumentParser(description="Train a notebook_06 style Tiny GPT")
    parser.add_argument("--data_path", type=str, default="data/input.txt")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find {data_path}. Put your training text there.")

    text = data_path.read_text(encoding="utf-8")
    if len(text) <= args.block_size + 1:
        raise ValueError("Training text is too short. Add more text to data/input.txt.")

    chars, stoi, itos = build_vocab(text)
    vocab_size = len(chars)
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

    dataset = NextTokenDataset(data, args.block_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyGPT(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print(f"device: {device}")
    print(f"text length: {len(text):,}")
    print(f"vocab size: {vocab_size}")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    for step in range(args.max_iters):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            break

        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            train_loss = estimate_loss(model, loader, device, args.eval_iters)
            print(f"step {step:4d} | train loss {train_loss:.4f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "tiny_gpt.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "block_size": args.block_size,
            "n_embd": args.n_embd,
            "n_head": args.n_head,
            "n_layer": args.n_layer,
            "dropout": args.dropout,
        },
        "stoi": stoi,
        "itos": itos,
    }, ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
