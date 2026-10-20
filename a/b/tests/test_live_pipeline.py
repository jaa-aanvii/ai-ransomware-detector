import os
import sys
import numpy as np

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from src.data_pipeline.parser import DatasetParser
from src.data_pipeline.window_engine import SlidingWindowEngine


DATA_PATH = "data/test_sample_concise.xlsx"


def test_parser():

    parser = DatasetParser(
        DATA_PATH
    )

    df = parser.load()

    assert len(df) > 0

    assert "pid" in df.columns

    assert "timestamp" in df.columns

    assert "label" in df.columns

    print("Parser test: PASSED")


def test_window_engine():

    engine = SlidingWindowEngine(
        sequence_length=6,
        feature_dimension=8
    )

    result = None

    for i in range(6):

        features = np.ones(
            8,
            dtype=np.float32
        ) * i

        result = engine.add_window(
            pid=1234,
            timestamp=i * 0.5,
            features=features,
            label=0
        )

    assert result is not None

    sequence, label = result

    assert sequence.shape == (6, 8)

    assert label == 0

    print("Window engine test: PASSED")


if __name__ == "__main__":

    test_parser()

    test_window_engine()

    print("\nAll tests passed.")