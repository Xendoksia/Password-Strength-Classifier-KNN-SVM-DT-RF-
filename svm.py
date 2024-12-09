import pandas as pd 

import numpy as np 

from sklearn.feature_extraction.text import TfidfVectorizer 

from sklearn.model_selection import train_test_split 

from sklearn.svm import SVC 

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, roc_curve 

import matplotlib.pyplot as plt 

import seaborn as sns 

import warnings 

warnings.filterwarnings("ignore") 

 

data = pd.read_csv("data.csv", on_bad_lines="skip") 

data = data.sample(n=100000, random_state=42) 

data = data.dropna() 

data.info() 

 

data['strength'] = data['strength'].map({0: "low", 1: "normal", 2: "high"}) 

 

password = np.array(data['password']) 

strength = np.array(data['strength']) 

 

def splitPassword(password): 

    return list(password) 

tfidf = TfidfVectorizer(analyzer=splitPassword) 

 

X = tfidf.fit_transform(password) 

 

x_train, x_test, y_train, y_test = train_test_split(X, strength, test_size=0.33, random_state=10) 

 

model = SVC(probability=True, kernel='linear')  # Use linear kernel, you can  

model.fit(x_train, y_train) 

 

y_pred = model.predict(x_test) 

y_test_encoded = pd.factorize(y_test)[0] 

y_pred_encoded = pd.factorize(y_pred, sort=True)[0] 

 

accuracy = accuracy_score(y_test, y_pred) 

conf_matrix = confusion_matrix(y_test, y_pred) 

precision = precision_score(y_test_encoded, y_pred_encoded, average="weighted") 

recall = recall_score(y_test_encoded, y_pred_encoded, average="weighted") 

f1 = f1_score(y_test_encoded, y_pred_encoded, average="weighted") 

 

print("Accuracy:", accuracy) 

print("Precision:", precision) 

print("Recall:", recall) 

print("F1 Score:", f1) 

 

plt.figure(figsize=(8, 6)) 

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['low', 'normal', 'high'], yticklabels=['low', 'normal', 'high']) 

plt.title("Confusion Matrix") 

plt.xlabel("Predicted") 

plt.ylabel("Actual") 

plt.tight_layout() 

 

plt.savefig("confusion_matrix_image_svm.png") 

plt.show() 

 

y_test_binarized = pd.get_dummies(pd.factorize(y_test)[0]) 

y_pred_proba = model.predict_proba(x_test) 

 

auc = roc_auc_score(y_test_binarized, y_pred_proba, average="weighted", multi_class="ovr") 

print("AUC-ROC:", auc) 

 

prediction_password = "PasswordStrenghtPrediction.123" 

prediction = tfidf.transform([prediction_password]).toarray() 

model_predict = model.predict(prediction) 

print("Predicted Strength:", model_predict[0]) 

 
