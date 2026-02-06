import torch
from torch import nn


class LOBTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 32,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 256,
    ):
        super().__init__()
        self.max_len = max_len
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.norm_in = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.pos_emb, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, input_dim]
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"seq_len={seq_len} is greater than max_len={self.max_len}")

        h = self.input_proj(x)
        h = h + self.pos_emb[:, :seq_len, :]
        h = self.norm_in(h)

        # Causal attention: each token only attends to past/current tokens.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
        )
        h = self.encoder(h, mask=causal_mask)

        # Predict using the last token representation.
        return self.head(h[:, -1, :])

