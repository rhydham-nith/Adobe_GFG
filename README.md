# Adobe_GFG
Welcome to the Adobe GFG Hackathon GitHub repository! This repository contains all the code, documentation, and resources related to our project developed during the Adobe GFG Hackathon

# Shape Detection with Bi-LSTM Models

This project aims to detect and recognize various geometric shapes from doodles using a Bi-LSTM (Bidirectional Long Short-Term Memory) model, a type of Recurrent Neural Network (RNN). Our approach is based on recent research, combined with our knowledge of data structures and algorithms.

## Table of Contents

- [About the Project](#about-the-project)
- [Bi-LSTM Model Overview](#bi-lstm-model-overview)
- [Our Approach and Methodology](#our-approach-and-methodology)
- [Future Plans and Approach](#future-plans-and-approach)
- [Challenges and Limitations](#challenges-and-limitations)
- [Acknowledgments](#acknowledgments)

## About the Project

The project is designed to detect and classify various shapes from test cases of doodling. We have developed individual models for each shape using a Bi-LSTM network, which are trained by running specific Python scripts provided in this repository. The models are then applied to test cases to identify and match the shapes present.


## Bi-LSTM Model Overview

### What is Bi-LSTM?

The Bi-LSTM (Bidirectional Long Short-Term Memory) is an advanced type of Recurrent Neural Network (RNN) that processes data in both forward and backward directions. This dual processing allows the model to capture dependencies in sequential data more effectively, making it well-suited for tasks like shape detection, where the order and context of points are critical.

### Why Bi-LSTM?

Bi-LSTM is particularly powerful for shape detection because:
- **Contextual Understanding:** It considers both previous and future points in the sequence, enabling a more comprehensive understanding of the shape.
- **Sequential Data Handling:** Shapes can be seen as sequences of points, and Bi-LSTM’s ability to handle sequential data makes it ideal for this task.
- **Improved Accuracy:** By processing data bidirectionally, Bi-LSTM models tend to be more accurate in detecting complex patterns and shapes.

## Our Approach and Methodology

Our process for shape detection and regularization involves the following steps:

1. **Training the Model:**  
   - We begin by training our Bi-LSTM models on various shapes that we have considered in the first phase. These shapes form the basis of our shape recognition system.

2. **Shape Detection in Test Cases:**  
   - Once the models are trained, we take the provided test cases and check if the shape is present in the dataset we used. If the shape is found, we proceed to the next step.

3. **Shape Regularization:**  
   - If the shape is detected in our dataset, we plot the shape and optimize it to produce a correct and smooth curve. This process is known as **regularization**, where the detected shape is refined and made more accurate.

4. **Future Enhancement - Recursive Backtracking Algorithm:**  
   - If given the opportunity to advance the project further, we plan to implement a recursive backtracking algorithm. This algorithm will identify shapes whose endpoints are close together and analyze them collectively to determine if they form a complete shape.

### Shape Detection

Once the models are trained, you can check every possible shape by iterating through them in a loop. The script will call each shape model and compare the results.

Example:
```python
shapes = ['circle', 'square', 'triangle']
for shape in shapes:
    # Load the corresponding model
    model = load_model(f'{shape}_model.h5')
    # Perform shape detection
    result = model.predict(data)
    print(f'Detected shape: {shape} with result: {result}')
```

## Future Plans and Approach

Our next steps involve refining the model's accuracy and expanding its capabilities. Here's how we plan to achieve that:

1. **Algorithm Enhancement:**
   - We plan to implement a recursion-based backtracking algorithm that identifies endpoints or specific figures that are close together. These figures will then be grouped to determine if they form a particular shape. This approach leverages our team's proficiency in Data Structures and Algorithms.

2. **Adaptive Learning:**
   - We will enhance the model to repetitively learn from new shapes that may emerge in the future. By continuously updating the model, we aim to improve its ability to recognize increasingly complex and varied shapes.

## Challenges and Limitations

Despite our efforts, we faced several challenges:

- **Lack of Practical Experience:** Our team had no prior hands-on experience with the Bi-LSTM model, which made the implementation process challenging.
- **Limited Guidance:** Unfortunately, we couldn't find any mentors or seniors in our institution who could guide us through this project.
- **Resource Constraints:** Given the resources available to us, we did our best to create and apply the model. With proper mentorship and additional resources, we believe we could significantly improve the model's performance and accuracy.

## Acknowledgments

- **Research Paper**: We based our approach on the methods outlined in Research Paper [Shape Recognition and Corner Points Detection in 2D Drawings Using a Machine Learning Long Short-Term Memory (LSTM) Approach]. The paper provided foundational knowledge for implementing the Bi-LSTM model in shape detection.
- **GFG Documentation**: The documentation on LSTM models from GeeksforGeeks was instrumental in helping us understand and apply this relatively new approach to our data.
- Special thanks to the open-source community for providing valuable resources and tools.
