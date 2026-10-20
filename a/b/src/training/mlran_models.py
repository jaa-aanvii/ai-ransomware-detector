import torch
import torch.nn as nn


class GRUFeatureExtractor(nn.Module):
    """
    GRU used only as a feature extractor.
    No classification/output layer.
    """

    def __init__(
        self,
        input_dim=483,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, x):
        # x: (batch, 1, 483)
        output, hidden = self.gru(x)

        # Last GRU hidden representation
        features = hidden[-1]

        return features


class LSTMFeatureExtractor(nn.Module):
    """
    LSTM used only as a feature extractor.
    No classification/output layer.
    """

    def __init__(
        self,
        input_dim=483,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, x):
        # x: (batch, 1, 483)
        output, (hidden, cell) = self.lstm(x)

        # Last LSTM hidden representation
        features = hidden[-1]

        return features


class BiLSTMFeatureExtractor(nn.Module):
    """
    Bidirectional LSTM used only as a feature extractor.
    No classification/output layer.
    """

    def __init__(
        self,
        input_dim=483,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

    def forward(self, x):
        # x: (batch, 1, 483)
        output, (hidden, cell) = self.lstm(x)

        # Last layer has two directions:
        # hidden[-2] = forward
        # hidden[-1] = backward

        forward_features = hidden[-2]
        backward_features = hidden[-1]

        # Combine both directions
        features = torch.cat(
            [forward_features, backward_features],
            dim=1
        )

        return features


class ParallelFeatureExtractor(nn.Module):
    """
    Runs GRU, LSTM and BiLSTM in parallel.

    Input:
        (batch, 1, 483)

    Output:
        fused feature representation
    """

    def __init__(
        self,
        input_dim=483,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.gru = GRUFeatureExtractor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.lstm = LSTMFeatureExtractor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.bilstm = BiLSTMFeatureExtractor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, x):

        # Three models process the SAME input
        gru_features = self.gru(x)
        lstm_features = self.lstm(x)
        bilstm_features = self.bilstm(x)

        # Combine learned representations
        fused_features = torch.cat(
            [
                gru_features,
                lstm_features,
                bilstm_features
            ],
            dim=1
        )

        return fused_features

    def extract_all(self, x):

        gru_features = self.gru(x)
        lstm_features = self.lstm(x)
        bilstm_features = self.bilstm(x)

        return (
            gru_features,
            lstm_features,
            bilstm_features
        )


if __name__ == "__main__":

    print("=" * 60)
    print("Testing Parallel GRU + LSTM + BiLSTM")
    print("=" * 60)

    # Simulate one batch of MLRan samples
    batch_size = 16
    input_features = 483

    x = torch.randn(
        batch_size,
        1,
        input_features
    )

    print(f"\nInput shape: {x.shape}")

    model = ParallelFeatureExtractor(
        input_dim=483,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    )

    model.eval()

    with torch.no_grad():

        gru_features, lstm_features, bilstm_features = (
            model.extract_all(x)
        )

        fused_features = model(x)

    print(f"GRU features    : {gru_features.shape}")
    print(f"LSTM features   : {lstm_features.shape}")
    print(f"BiLSTM features : {bilstm_features.shape}")
    print(f"Fused features  : {fused_features.shape}")

    print("\nExpected:")
    print("GRU    -> (16, 128)")
    print("LSTM   -> (16, 128)")
    print("BiLSTM -> (16, 256)")
    print("Fused  -> (16, 512)")

    print("\nModel test completed successfully.")
