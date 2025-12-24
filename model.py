# model.py
import torch
from torch import nn


class GruLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hdn_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        tie_wght: bool = False,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hdn_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.out = nn.Linear(hdn_dim, vocab_size, bias=False)

        # Weight tying only valid if dimensions match
        if tie_wght:
            if emb_dim != hdn_dim:
                raise ValueError("Weight tying requires emb_dim == hdn_dim")
            self.out.weight = self.embedding.weight

    def forward(self, input_ids: torch.Tensor):
        x = self.embedding(input_ids)          # (1, T, E)
        out, hidden = self.gru(x)              # (1, T, H)
        logits = self.out(out)                 # (1, T, V)
        return logits, hidden
