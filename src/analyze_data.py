from datasets import load_dataset
import pandas as pd

# Load IMDb dataset
dataset = load_dataset("imdb")

# Convert training data to pandas
train_df = pd.DataFrame(dataset["train"])

# Convert test data to pandas
test_df = pd.DataFrame(dataset["test"])

print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))

print("\nTraining dataset:")
print(train_df.head())

print("\nLabel distribution:")
print(train_df["label"].value_counts())

print("\nDataset information:")
print(train_df.info())