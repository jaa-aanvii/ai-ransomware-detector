import torch
import torch.nn as nn

class RansomwareLSTM(nn.Module):
    def __init__(self, num_features: int = 10, hidden_dim: int = 32, num_layers: int = 1):
        """
        Lightweight LSTM for OS process telemetry sequence classification.
        Input Tensor Shape: (Batch Size, Sequence Length, Num Features) -> (B, T, F)
        Output Tensor Shape: (Batch Size, 1) -> Threat Risk Score (0.0 to 1.0)
        """
        super(RansomwareLSTM, self).__init__()
        
        # Recurrent layer to capture behavioral evolution across 500ms windows
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True  # Guarantees input shape is (B, T, F)
        )
        
        # Classification head to output threat probability
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # lstm_out shape: (B, T, hidden_dim)
        # _ contains (hn, cn) hidden states
        lstm_out, _ = self.lstm(x)
        
        # Extract the hidden state of the LAST time step (T_final)
        last_step_out = lstm_out[:, -1, :]
        
        # Map to threat risk score
        logits = self.fc(last_step_out)
        risk_score = self.sigmoid(logits)
        return risk_score