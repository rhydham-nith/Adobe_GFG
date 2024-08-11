import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from non_data import generate_non_circle
radius = 1
num_points = 100
def generate_circle(radius, num_points):
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.array([x, y]).T

def generate_distorted_circle(radius, num_points, distortion_level=0.1):
    theta = np.linspace(0, 2 * np.pi, num_points)

    # Introduce random distortion in radius
    distortions = np.random.uniform(1 - distortion_level, 1 + distortion_level, num_points)
    x = radius * distortions * np.cos(theta)
    y = radius * distortions * np.sin(theta)

    # Optionally, apply an affine transformation to stretch or rotate
    transform_matrix = np.array([[np.random.uniform(0.9, 1.1), np.random.uniform(-0.1, 0.1)],
                                 [np.random.uniform(-0.1, 0.1), np.random.uniform(0.9, 1.1)]])

    distorted_circle = np.dot(np.array([x, y]).T, transform_matrix)
    return distorted_circle

# Generate multiple examples of circles and non-circles
num_samples = 1000
circle_samples = [generate_circle(radius, num_points) for _ in range(num_samples)]
non_circle_samples = [generate_non_circle(num_points) for _ in range(num_samples)]

# Labels: 1 for circle, 0 for non-circle
labels = np.array([1] * num_samples + [0] * num_samples)

# Combine the data
data = np.array(circle_samples + non_circle_samples)

# Shuffle the dataset
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
# Generate additional distorted circle samples
distorted_circle_samples = [generate_distorted_circle(radius, num_points) for _ in range(num_samples)]

# Combine all data
augmented_data = np.array(circle_samples + non_circle_samples + distorted_circle_samples)
augmented_labels = np.array([1] * num_samples + [0] * num_samples + [1] * num_samples)

# Shuffle the augmented dataset
indices = np.arange(augmented_data.shape[0])
np.random.shuffle(indices)
augmented_data = augmented_data[indices]
augmented_labels = augmented_labels[indices]
# Split the augmented dataset
X_train_aug, X_test_aug, y_train_aug, y_test_aug = train_test_split(augmented_data, augmented_labels, test_size=0.2, random_state=42)

# Define and compile the Bi-LSTM model
model_circle = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(num_points, 2)),
    Bidirectional(LSTM(64)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_circle.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with the augmented dataset
history = model_circle.fit(X_train_aug, y_train_aug, epochs=5, batch_size=32, validation_data=(X_test_aug, y_test_aug))

import pickle

with open('model_circle.pkl', 'wb') as file:
    pickle.dump(model_circle, file)