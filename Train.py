import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the directory paths for the images
train_clear_images_dir = "TrainingSet/Undistorted"
train_blur_images_dir1 = "TrainingSet/Naturally-Blurred"
train_blur_images_dir2 = "TrainingSet/Artificially-Blurred"

# Define a function to process the Laplacian of an image
"""def process_laplacian(image):
    laplacian = np.array(image.filter(ImageFilter.LAPLACIAN))
    return laplacian"""

def process_laplacian(image):
    image = image.resize((600,400))
    image = np.asarray(image)
    image = image / 255.0
    return cv2.Laplacian(image, cv2.CV_64F)

# Process the clear images in the training directory
clear_img_max_laplacian = []
clear_img_var_laplacian = []
for i in os.listdir(train_clear_images_dir):
    image = Image.open(os.path.join(train_clear_images_dir, i)).convert('L')
    laplacian = process_laplacian(image)
    clear_img_max_laplacian.append(laplacian.max())
    clear_img_var_laplacian.append(laplacian.var())

# Process the blurred images in the training directory
blurred_img_max_laplacian = []
blurred_img_var_laplacian = []
for i in os.listdir(train_blur_images_dir1):
    image = Image.open(os.path.join(train_blur_images_dir1, i)).convert('L')
    laplacian = process_laplacian(image)
    blurred_img_max_laplacian.append(laplacian.max())
    blurred_img_var_laplacian.append(laplacian.var())
for i in os.listdir(train_blur_images_dir2):
    image = Image.open(os.path.join(train_blur_images_dir2, i)).convert('L')
    laplacian = process_laplacian(image)
    blurred_img_max_laplacian.append(laplacian.max())
    blurred_img_var_laplacian.append(laplacian.var())

# Combine the data into a single dataframe
labels = np.append(np.zeros(len(clear_img_max_laplacian)), np.ones(len(blurred_img_max_laplacian)))
laplacian_max = clear_img_max_laplacian + blurred_img_max_laplacian
laplacian_var = clear_img_var_laplacian + blurred_img_var_laplacian
train_data = pd.DataFrame({
    'Laplacian_Max': laplacian_max,
    'Laplacian_Var': laplacian_var,
    'Label': labels
})

# Shuffle and split the data into training and validation sets
train_data = train_data.sample(frac=1).reset_index(drop=True)
X = train_data[['Laplacian_Max', 'Laplacian_Var']]
y = train_data['Label']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the test set data to a CSV file
test_data = pd.DataFrame({'Laplacian_Max': X_val['Laplacian_Max'], 'Laplacian_Var': X_val['Laplacian_Var'], 'Label': y_val})
test_data.to_csv('TestSet.csv', index=False)

'''# Define the model architecture
model = Sequential()
model.add(Dense(128, input_shape=(2,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))'''

# Create the model
model = Sequential()
model.add(Dense(64, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Save the model
model.save('model.h5')