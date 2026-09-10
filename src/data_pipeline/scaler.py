import os
import joblib

from sklearn.preprocessing import StandardScaler


class FeatureScaler:

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X):
        self.scaler.fit(X)
        return self

    def transform(self, X):
        return self.scaler.transform(X)

    def fit_transform(self, X):
        return self.scaler.fit_transform(X)

    def save(self, path):
        directory = os.path.dirname(path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        joblib.dump(self.scaler, path)

    def load(self, path):
        self.scaler = joblib.load(path)
        return self