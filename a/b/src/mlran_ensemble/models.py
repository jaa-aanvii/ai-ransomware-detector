"""
models.py -- LSTM, GRU, and BiLSTM encoders.

Each model can run in two modes:

  * return_embedding=False (default): full classifier. Runs the
    recurrent encoder, then a Linear "output layer" down to a single
    logit. Used while training/tuning each model individually, so each
    one actually learns a useful representation on its own.

  * return_embedding=True: the output layer is skipped entirely and the
    pooled hidden representation is returned instead. This is the mode
    used for the ensemble step -- "remove the output layer of the three
    models and feed their features into the fine-tuning model".

All three share the same embedding size (hidden_dim, or 2*hidden_dim
for the bidirectional model) so their embeddings concatenate cleanly.
"""

import torch
import torch.nn as nn


class _BaseRNNClassifier(nn.Module):
    """Shared forward-pass logic for LSTM/GRU: run the recurrent layer,
    take the last timestep's hidden state as the pooled embedding, and
    optionally project it down to a single logit."""

    def __init__(self, embedding_dim, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(embedding_dim, 1)

    def _pool(self, rnn_output):
        # rnn_output: (batch, seq_len, embedding_dim) -> last timestep
        return rnn_output[:, -1, :]

    def _finish(self, embedding, return_embedding):
        embedding = self.dropout(embedding)

        if return_embedding:
            return embedding

        return self.output_layer(embedding)


class LSTMEncoder(_BaseRNNClassifier):
    def __init__(self, input_size, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__(embedding_dim=hidden_dim, dropout=dropout)
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x, return_embedding=False):
        out, _ = self.rnn(x)
        embedding = self._pool(out)
        return self._finish(embedding, return_embedding)

    @property
    def embedding_dim(self):
        return self.output_layer.in_features


class GRUEncoder(_BaseRNNClassifier):
    def __init__(self, input_size, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__(embedding_dim=hidden_dim, dropout=dropout)
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x, return_embedding=False):
        out, _ = self.rnn(x)
        embedding = self._pool(out)
        return self._finish(embedding, return_embedding)

    @property
    def embedding_dim(self):
        return self.output_layer.in_features


class BiLSTMEncoder(_BaseRNNClassifier):
    def __init__(self, input_size, hidden_dim=64, num_layers=1, dropout=0.2):
        # Bidirectional doubles the embedding size (forward + backward).
        super().__init__(embedding_dim=hidden_dim * 2, dropout=dropout)
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x, return_embedding=False):
        out, _ = self.rnn(x)
        embedding = self._pool(out)
        return self._finish(embedding, return_embedding)

    @property
    def embedding_dim(self):
        return self.output_layer.in_features


MODEL_REGISTRY = {
    "lstm": LSTMEncoder,
    "gru": GRUEncoder,
    "bilstm": BiLSTMEncoder,
}
