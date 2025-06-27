# Logistic Regression on Breast Cancer dataset 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn imports — keeping them grouped for clean , readable readings 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, precision_score, recall_score
)

# Loading the dataset - built-in breast cancer set
cancer = load_breast_cancer()

# Dumping into a DataFrame for convenience
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

# Saving
df.to_csv('cleaned_breast_cancer.csv', index=False)

# defining  predictors and target variable
features = df.drop('target', axis=1)
target = df['target']

# Data split (80/20) — random_state fixed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Scaling — standard scaling helps for Logistic Regression (LR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fitting
X_test_scaled = scaler.transform(X_test)        # only transform test

# training the logistic regression model
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

# Making the predictions (Class and Probabilities)
predicted_labels = log_model.predict(X_test_scaled)
predicted_probs = log_model.predict_proba(X_test_scaled)[:, 1]  # probabilities for class 1

# Evaluation of the metrics
conf_matrix = confusion_matrix(y_test, predicted_labels)
prec_score = precision_score(y_test, predicted_labels)
rec_score = recall_score(y_test, predicted_labels)
roc_auc_val = roc_auc_score(y_test, predicted_probs)

# Output evaluations
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y_test, predicted_labels))
print("ROC-AUC Score:", roc_auc_val)

# Visual: Confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()


# Visual: ROC curve
fpr_vals, tpr_vals, _ = roc_curve(y_test, predicted_probs)
plt.figure(figsize=(6, 4))
plt.plot(fpr_vals, tpr_vals, label=f'ROC curve (AUC = {roc_auc_val:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()


# Manually tweaking the threshold 
manual_thresh = 0.3
adjusted_preds = (predicted_probs > manual_thresh).astype(int)

print(f"\nUsing Adjusted Threshold = {manual_thresh}")
print("Precision (adjusted):", precision_score(y_test, adjusted_preds))
print("Recall (adjusted):", recall_score(y_test, adjusted_preds))

# Plotting the sigmoid function for reference 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))  #  logic behind logistic regression

z_vals = np.linspace(-10, 10, 100)
plt.figure(figsize=(6, 4))
plt.plot(z_vals, sigmoid(z_vals))
plt.title("Sigmoid Activation Curve")
plt.xlabel("Input (z)")
plt.ylabel("Sigmoid(z)")
plt.grid(True)
plt.tight_layout()
plt.show()
