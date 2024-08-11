import matplotlib.pyplot as plt
import numpy as np
def generate_non_circle(num_points):
    x = np.random.uniform(-1, 1, num_points)
    y = np.random.uniform(-1, 1, num_points)
    return np.array([x, y]).T
non_circle_data = generate_non_circle(100)
