from torch import nn


class SensitiveClassifierByMiniLM(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.classifier(x).squeeze(-1)
