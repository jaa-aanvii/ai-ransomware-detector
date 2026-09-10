import torch
import torch.nn as nn


class RansomwareLSTM(nn.Module):
    def __init__(
        self,
        num_features=8,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # Take output from the final time step
        last_step = lstm_out[:, -1, :]

        last_step = self.dropout(last_step)

        logits = self.fc(last_step)

        return logits