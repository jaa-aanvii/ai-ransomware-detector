import torch

from src.training.model import RansomwareLSTM


def test_forward_pass_output_shape():
    model = RansomwareLSTM(num_features=8, hidden_dim=32, num_layers=1)
    batch = torch.randn(4, 6, 8)  # (B=4, T=6, F=8)

    output = model(batch)

    assert output.shape == (4, 1)


def test_forward_pass_single_sample():
    model = RansomwareLSTM(num_features=8, hidden_dim=32, num_layers=1)
    single = torch.randn(1, 6, 8)

    output = model(single)

    assert output.shape == (1, 1)


def test_output_is_finite_logit():
    model = RansomwareLSTM(num_features=8, hidden_dim=32, num_layers=1)
    model.eval()
    batch = torch.randn(2, 6, 8)

    with torch.no_grad():
        output = model(batch)
        probability = torch.sigmoid(output)

    assert torch.isfinite(output).all()
    assert (probability >= 0).all() and (probability <= 1).all()


def test_rejects_wrong_sequence_length_gracefully_or_runs():
    # LSTM itself is sequence-length agnostic (it's the surrounding
    # pipeline -- window_engine, dataset.py -- that enforces length 6),
    # so a different T should still produce a valid (B, 1) output here.
    model = RansomwareLSTM(num_features=8, hidden_dim=32, num_layers=1)
    batch = torch.randn(3, 10, 8)

    output = model(batch)

    assert output.shape == (3, 1)
