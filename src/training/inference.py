import torch

from .model import RansomwareLSTM


class RansomwareInference:

    def __init__(
        self,
        model_path,
        device="cpu"
    ):

        self.device = torch.device(device)

        self.model = RansomwareLSTM(
            num_features=8
        )

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device
            )
        )

        self.model.to(self.device)

        self.model.eval()

    def predict(self, sequence):

        x = torch.tensor(
            sequence,
            dtype=torch.float32
        )

        if x.ndim == 2:
            x = x.unsqueeze(0)

        x = x.to(self.device)

        with torch.no_grad():

            output = self.model(x)

            probability = torch.sigmoid(
                output
            ).item()

        label = (
            "RANSOMWARE"
            if probability >= 0.5
            else "BENIGN"
        )

        return {
            "probability": probability,
            "label": label
        }