import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
from non_data import generate_non_circle

def generate_decagon_points(num_points=100):
  # Calculate the coordinates of the decagon vertices
  angle = 2 * np.pi / 10
  vertices = np.array([[np.cos(i * angle), np.sin(i * angle)] for i in range(10)])

  # Distribute points evenly along each side
  points_per_side = num_points // 10
  remaining_points = num_points % 10

  all_points = []
  for i in range(10):
    start = vertices[i]
    end = vertices[(i + 1) % 10]
    side_length = np.linalg.norm(end - start)
    step = side_length / (points_per_side + 1)

    for j in range(points_per_side):
      point = start + (j + 1) * step * (end - start) / side_length
      all_points.append(point)

  # Add remaining points to the first side
  for i in range(remaining_points):
    start = vertices[0]
    end = vertices[1]
    side_length = np.linalg.norm(end - start)
    step = side_length / (num_points + 1)
    point = start + (points_per_side + i + 1) * step * (end - start) / side_length
    all_points.append(point)

  return np.array(all_points)


def generate_distorted_decagon_points(num_points=100, distortion_level=0.1):
  # Calculate the coordinates of the decagon vertices
  angle = 2 * np.pi / 10
  vertices = np.array([[np.cos(i * angle), np.sin(i * angle)] for i in range(10)])

  # Distribute points evenly along each side
  points_per_side = num_points // 10
  remaining_points = num_points % 10

  all_points = []
  for i in range(10):
    start = vertices[i]
    end = vertices[(i + 1) % 10]
    side_length = np.linalg.norm(end - start)
    step = side_length / (points_per_side + 1)

    for j in range(points_per_side):
      point = start + (j + 1) * step * (end - start) / side_length
      all_points.append(point)

  # Add remaining points to the first side
  for i in range(remaining_points):
    start = vertices[0]
    end = vertices[1]
    side_length = np.linalg.norm(end - start)
    step = side_length / (num_points + 1)
    point = start + (points_per_side + i + 1) * step * (end - start) / side_length
    all_points.append(point)

  # Introduce distortion
  all_points = np.array(all_points)
  distortion = np.random.uniform(-distortion_level/2, distortion_level/2, size=all_points.shape)
  distorted_points = all_points + distortion

  return distorted_points

distorted_points = generate_distorted_decagon_points(100, 0.1)

num_samples = 1000
samples = [generate_distorted_decagon_points() for _ in range(num_samples)]
non_samples = [generate_non_circle(100) for _ in range(num_samples)]

# Labels: 1 for circle, 0 for non-circle
labels = np.array([1] * num_samples + [0] * num_samples)

# Combine the data
data = np.array(samples + non_samples)

# Shuffle the dataset
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
# Generate additional distorted circle samples
distorted_samples = [generate_distorted_decagon_points(100, 0.1) for _ in range(num_samples)]

# Combine all data
augmented_data = np.array(samples + non_samples + distorted_samples)
augmented_labels = np.array([1] * num_samples + [0] * num_samples + [1] * num_samples)

# Shuffle the augmented dataset
indices = np.arange(augmented_data.shape[0])
np.random.shuffle(indices)
augmented_data = augmented_data[indices]
augmented_labels = augmented_labels[indices]
# Split the augmented dataset
X_train_aug, X_test_aug, y_train_aug, y_test_aug = train_test_split(augmented_data, augmented_labels, test_size=0.2, random_state=42)

# Define and compile the Bi-LSTM model
model_decagon = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(100, 2)),
    Bidirectional(LSTM(64)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_decagon.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with the augmented dataset
history = model_decagon.fit(X_train_aug, y_train_aug, epochs=4, batch_size=32, validation_data=(X_test_aug, y_test_aug))

import pickle

with open('model_decagon.pkl', 'wb') as file:
    pickle.dump(model_decagon, file)