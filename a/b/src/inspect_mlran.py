import pandas as pd

folder = "a/b/data/MLRan/"
labels = pd.read_csv(folder + "MLRan_labels.csv")
train_ids = pd.read_csv(folder + "type_train_sampleids.csv")
test_ids = pd.read_csv(folder + "type_test_sampleids.csv")

print("Labels:", labels.shape, labels.columns.tolist())
print(labels.head())

print("\nTrain IDs:", train_ids.shape, train_ids.columns.tolist())
print(train_ids.head())

print("\nTest IDs:", test_ids.shape, test_ids.columns.tolist())
print(test_ids.head())