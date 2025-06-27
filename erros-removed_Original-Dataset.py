from sklearn.datasets import load_breast_cancer
import pandas as pd

#  breast cancer dataset for testing
dataset = load_breast_cancer()

# Converting to a DataFrame so it is easier to work with in Pandas
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

# Adding the labels (malignant/benign), calling it ‘target’ like usual
df['target'] = dataset.target

# Saving the cleaned version 
csv_file_path = 'errors_removed_OriginalDataset.csv' 
df.to_csv(csv_file_path, index=False)

# Confirmation 
print(" Saved clean original dataset as errors_removed_OriginalDataset.csv")


