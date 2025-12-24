# train.py
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from model import GruLM
from dataloader import DataLoader

import sentencepiece as spm


def train(
    model: GruLM,
    tokens: torch.Tensor,
    pad_id: int,
    epochs: int = 5,
    lr: float = 3e-4,
    grad_clip: float = 1.0,
):

    assert tokens.dim() == 2, "Expected (N, T) token tensor"

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    print("\n========== DATA DEBUG ==========")
    print(f"Sequences  : {tokens.size(0)}")
    print(f"Seq length : {tokens.size(1)}")
    print(f"Pad ratio  : {(tokens == pad_id).float().mean():.4f}")
    print("================================\n")

    for epoch in range(epochs):
        total_loss = 0.0
        total_tokens = 0

        for i in range(tokens.size(0)):
            seq = tokens[i:i+1]           # (1, T)
            x = seq[:, :-1]               # (1, T-1)
            y = seq[:, 1:]                # (1, T-1)

            optimizer.zero_grad(set_to_none=True)

            logits, _ = model(x)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=pad_id,
            )

            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            valid = (y != pad_id).sum().item()
            total_loss += loss.item() * valid
            total_tokens += valid

            if i % 50 == 0:
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    acc = (
                        (preds == y)
                        .masked_select(y != pad_id)
                        .float()
                        .mean()
                        .item()
                    )

                print(
                    f"[Epoch {epoch+1}] "
                    f"Sample {i}/{tokens.size(0)} | "
                    f"Loss {loss.item():.4f} | "
                    f"TokenAcc {acc:.3f}"
                )

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = torch.exp(torch.tensor(avg_loss)).item()

        print(
            f"\n====== Epoch {epoch+1} Summary ======\n"
            f"Avg NLL     : {avg_loss:.4f}\n"
            f"Perplexity : {ppl:.2f}\n"
            f"Tokens     : {total_tokens}\n"
            f"===================================\n"
        )

    return avg_loss

# ---------------------------------------------
# Load tokenizer
# ---------------------------------------------
sp = spm.SentencePieceProcessor()
sp.load("tokenizer.model")

pad_id = sp.pad_id()
vocab_size = sp.get_piece_size()

print(f"Vocab size : {vocab_size}")
print(f"Pad id     : {pad_id}")

# ---------------------------------------------
# Load dataset (already tokenized + padded)
# ---------------------------------------------
dataset = DataLoader("out.txt", "tokenizer.model")
tokens = dataset.get_with_pad() 
tokens = torch.tensor(tokens, dtype=torch.long)

# ---------------------------------------------
# Init model (CPU only)
# ---------------------------------------------
model = GruLM(
    vocab_size=vocab_size,
    emb_dim=200,
    hdn_dim=200,
    num_layers=2,
    dropout=0.1,
    tie_wght=True,
)

# ---------------------------------------------
# Train
# ---------------------------------------------
train(
    model=model,
    tokens=tokens,
    pad_id=pad_id,
    epochs=4,
    lr=3e-4,
)

torch.save(model.state_dict(),'grulm.pt')
