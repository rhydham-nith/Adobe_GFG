import numpy as np
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from non_data import generate_non_circle  
width=2
height=1
center=[0,0] 
num_points=100
def generate_rectangle(width, height, center, num_points=100,distortion_level=0.1):
    half_width = width / 2
    half_height = height / 2
    x = []
    y = []

    # Divide the points equally among the four sides of the rectangle
    points_per_side = num_points // 4

    # Bottom side
    x.extend(np.linspace(center[0] - half_width, center[0] + half_width, points_per_side))
    y.extend([center[1] - half_height] * points_per_side)

    # Right side
    x.extend([center[0] + half_width] * points_per_side)
    y.extend(np.linspace(center[1] - half_height, center[1] + half_height, points_per_side))

    # Top side
    x.extend(np.linspace(center[0] + half_width, center[0] - half_width, points_per_side))
    y.extend([center[1] + half_height] * points_per_side)

    # Left side
    x.extend([center[0] - half_width] * points_per_side)
    y.extend(np.linspace(center[1] + half_height, center[1] - half_height, points_per_side))

    return np.column_stack((x, y))

rectangle_points = generate_rectangle(2, 1, [0, 0], 100)


def generate_distorted_rectangle(width, height, center, num_points=100, distortion_level=0.1):
    half_width = width / 2
    half_height = height / 2
    x = []
    y = []

    # Divide the points equally among the four sides of the rectangle
    points_per_side = num_points // 4

    # Bottom side
    x.extend(np.linspace(center[0] - half_width, center[0] + half_width, points_per_side))
    y.extend([center[1] - half_height] * points_per_side)

    # Right side
    x.extend([center[0] + half_width] * points_per_side)
    y.extend(np.linspace(center[1] - half_height, center[1] + half_height, points_per_side))

    # Top side
    x.extend(np.linspace(center[0] + half_width, center[0] - half_width, points_per_side))
    y.extend([center[1] + half_height] * points_per_side)

    # Left side
    x.extend([center[0] - half_width] * points_per_side)
    y.extend(np.linspace(center[1] + half_height, center[1] - half_height, points_per_side))

    points = np.column_stack((x, y))

    # Introduce distortion
    distortion = np.random.uniform(-distortion_level, distortion_level, size=points.shape)
    distorted_points = points + distortion

    return distorted_points

distorted_rectangle_points = generate_distorted_rectangle(2, 1, [0, 0], 100, 0.1)

num_samples = 1000
rectangle_samples = [generate_rectangle(width, height, center, num_points=100, distortion_level=0.1) for _ in range(num_samples)]
non_circle_samples = [generate_non_circle(num_points) for _ in range(num_samples)]

# Labels: 1 for circle, 0 for non-circle
labels = np.array([1] * num_samples + [0] * num_samples)

# Combine the data
data = np.array(rectangle_samples + non_circle_samples)

# Shuffle the dataset
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
# Generate additional distorted circle samples
distorted_rectangle_samples = [generate_distorted_rectangle(width, height, center, num_points=100, distortion_level=0.1) for _ in range(num_samples)]

# Combine all data
augmented_data = np.array(rectangle_samples + non_circle_samples + distorted_rectangle_samples)
augmented_labels = np.array([1] * num_samples + [0] * num_samples + [1] * num_samples)

# Shuffle the augmented dataset
indices = np.arange(augmented_data.shape[0])
np.random.shuffle(indices)
augmented_data = augmented_data[indices]
augmented_labels = augmented_labels[indices]
# Split the augmented dataset
X_train_aug, X_test_aug, y_train_aug, y_test_aug = train_test_split(augmented_data, augmented_labels, test_size=0.2, random_state=42)

# Define and compile the Bi-LSTM model
model_rectangle = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(num_points, 2)),
    Bidirectional(LSTM(64)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_rectangle.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with the augmented dataset
history = model_rectangle.fit(X_train_aug, y_train_aug, epochs=5, batch_size=32, validation_data=(X_test_aug, y_test_aug))

import pickle

with open('model_rectangle.pkl', 'wb') as file:
    pickle.dump(model_rectangle, file)