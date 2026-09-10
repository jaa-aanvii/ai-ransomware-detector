import torch

from .model import RansomwareLSTM
from src.data_pipeline.scaler import FeatureScaler


class RansomwareInference:

    def __init__(
        self,
        model_path,
        scaler_path="models/scaler.pkl",
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

        self.scaler = FeatureScaler()
        self.scaler.load(scaler_path)

    def predict(self, sequence):

        # sequence shape: (6, 8)
        x = torch.tensor(
            sequence,
            dtype=torch.float32
        )

        if x.ndim == 2:
            x = x.unsqueeze(0)

        # Scale using the SAME scaler used during training
        original_shape = x.shape

        x_np = x.cpu().numpy()
        x_np = x_np.reshape(-1, 8)

        x_np = self.scaler.transform(x_np)

        x_np = x_np.reshape(original_shape)

        x = torch.tensor(
            x_np,
            dtype=torch.float32,
            device=self.device
        )

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