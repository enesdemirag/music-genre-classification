# In this script Support Vector Machine (SVM) algorithm used on dataset.csv
# This dataset consist of 1000 samples with 26 features each
# https://scikit-learn.org/stable/modules/svm.html

import numpy as np
from utils import load_analytic_data, save_sklearn_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

dataset = load_analytic_data("dataset.csv")

# Encoding the labels
genres = dataset.iloc[:, -1] # Last column
encoder = LabelEncoder()
labels = encoder.fit_transform(genres)

# Scaling the features
scaler = StandardScaler() # MinMaxScaler() can be also used
features = scaler.fit_transform(np.array(dataset.iloc[:, :-1], dtype=float))

# Dividing dataset into training and testing sets
# 80to20 split
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Create knn model
model = SVC() #  NuSVC, SVR, NuSVR, LinearSVC, LinearSVR and OneClassSVM

# Training 
model.fit(x_train, y_train)

# Testing
accuracy = model.score(x_test, y_test)
print(accuracy)

# Save model
save_sklearn_model(model, "svm.sk")