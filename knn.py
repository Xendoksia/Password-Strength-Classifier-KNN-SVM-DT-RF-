import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Load and preprocess data
data = pd.read_csv("data.csv", on_bad_lines="skip")
data = data.sample(n=100000, random_state=42)
data = data.dropna()
data.info()

# Map strength values
data['strength'] = data['strength'].map({0: "low", 1: "normal", 2: "high"})

# Convert to arrays
password = np.array(data['password'])
strength = np.array(data['strength'])

# Define a preprocessing function
def splitPassword(password):
    return list(password)

# Initialize TfidfVectorizer
tfidf = TfidfVectorizer(analyzer=splitPassword)

# Transform passwords
X = tfidf.fit_transform(password)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(X, strength, test_size=0.33, random_state=10)

# Train a K-Nearest Neighbors model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

# Evaluate the model
y_pred = model.predict(x_test)
y_test_encoded = pd.factorize(y_test)[0]
y_pred_encoded = pd.factorize(y_pred, sort=True)[0]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test_encoded, y_pred_encoded, average="weighted")
recall = recall_score(y_test_encoded, y_pred_encoded, average="weighted")
f1 = f1_score(y_test_encoded, y_pred_encoded, average="weighted")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Plot confusion matrix as an image using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['low', 'normal', 'high'], yticklabels=['low', 'normal', 'high'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

# Save the confusion matrix image
plt.savefig("confusion_matrix_image_knn.png")
plt.show()

# ROC-AUC (One-vs-Rest for multiclass)
y_test_binarized = pd.get_dummies(pd.factorize(y_test)[0])
y_pred_proba = model.predict_proba(x_test)

auc = roc_auc_score(y_test_binarized, y_pred_proba, average="weighted", multi_class="ovr")
print("AUC-ROC:", auc)


prediction_password = "PasswordStrenghtPrediction.123"
prediction = tfidf.transform([prediction_password]).toarray()
model_predict = model.predict(prediction)
print("Predicted Strength:", model_predict[0])
