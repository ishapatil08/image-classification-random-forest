import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

Code to Load Images
data = []
labels = []

categories = ['cats', 'dogs']
dataset_path = 'dataset'

for category in categories:
    path = os.path.join(dataset_path, category)
    class_label = categories.index(category)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (64, 64))
        data.append(image.flatten())   # Convert image to 1D
        labels.append(class_label)

data = np.array(data)
labels = np.array(labels)

print("Data shape:", data.shape)

Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)
Train Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

Prediction
y_pred = rf_model.predict(X_test)
🔹 Step 6: Evaluate Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
