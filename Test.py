import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from PIL import Image, ImageFilter
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("model.h5")

test_data = pd.read_csv("TestSet.csv")
test_data = test_data.sample(frac=1).reset_index(drop=True)

X = test_data.iloc[:, :2]
y = test_data.iloc[:, 2]

prediction = model.predict(X)

print(prediction)