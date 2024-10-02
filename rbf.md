In this project, you implemented an **RBF Network** to classify the MNIST dataset, which consists of hand-written digit images. The architecture involves using Radial Basis Functions (RBF) to map input data into a high-dimensional space, and a linear model to classify the transformed data.

Let's break down the mathematical components of your RBF network and map them to the implementation:

### 1. **Dataset Preparation**:
You are working with the **MNIST** dataset, which contains 70,000 images (28x28 pixels) of handwritten digits (0–9). The goal is to classify each image into one of 10 digit classes.

- **Input** \(X\): Each image is represented as a vector of 784 pixel values (28x28 pixels flattened). After normalization (scaling between 0 and 1), each input data point \( \mathbf{x}_i \in \mathbb{R}^{784} \).

- **Target** \( y \): Each target \( y_i \in \{0, 1, ..., 9\} \) represents the digit label.

Mathematically, the dataset can be written as:
- \( X = [\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_n] \in \mathbb{R}^{n \times 784} \), where \(n\) is the number of samples.
- \( y = [y_1, y_2, ..., y_n] \in \mathbb{R}^{n} \), where \( y_i \in \{0, 1, ..., 9\} \).

### 2. **Radial Basis Functions**:
The key feature of an RBF network is the **RBF layer**, which transforms the input data. An RBF \( \phi(\mathbf{x}, \mathbf{c}) \) is a function that depends on the distance between the input \( \mathbf{x} \) and a center \( \mathbf{c} \).

#### Radial Basis Function Formulation:
You implemented several different types of RBFs. The general form for any RBF is:
\[ \phi(\mathbf{x}, \mathbf{c}) = f(|| \mathbf{x} - \mathbf{c} ||) \]
Where:
- \( \mathbf{x} \) is an input vector.
- \( \mathbf{c} \) is the center of the RBF.
- \( || \mathbf{x} - \mathbf{c} || \) is the Euclidean distance between \( \mathbf{x} \) and \( \mathbf{c} \).

#### Types of RBFs:
1. **Gaussian RBF**: 
   \[
   \phi(\mathbf{x}, \mathbf{c}, \sigma) = \exp\left(-\frac{||\mathbf{x} - \mathbf{c}||^2}{2\sigma^2}\right)
   \]
   This is a Gaussian (bell-shaped) function, where \( \sigma \) controls the width of the curve.
   
2. **Multiquadric RBF**: 
   \[
   \phi(\mathbf{x}, \mathbf{c}) = \sqrt{1 + ||\mathbf{x} - \mathbf{c}||^2}
   \]
   This function grows as the distance increases.

3. **Inverse Multiquadric RBF**: 
   \[
   \phi(\mathbf{x}, \mathbf{c}) = \frac{1}{\sqrt{1 + ||\mathbf{x} - \mathbf{c}||^2}}
   \]
   This function decays as the distance increases.

4. **Thin-Plate Spline RBF**:
   \[
   \phi(\mathbf{x}, \mathbf{c}) = ||\mathbf{x} - \mathbf{c}||^2 \log(||\mathbf{x} - \mathbf{c}||)
   \]
   This function grows logarithmically with distance.

### 3. **RBF Centers** (Clustering):
Before applying the RBF transformation, the network selects **centers** for the RBFs. These centers can be thought of as prototypes that are representative of the data.

- You used **K-Means** to find \( k \) clusters in the input data. These cluster centers are the **RBF centers** \( \mathbf{C} \).
  
Mathematically, you partition the dataset into \( k \) clusters, with each center \( \mathbf{c}_i \) as the centroid of the \( i \)-th cluster. 
The centers \( \mathbf{C} = [\mathbf{c}_1, \mathbf{c}_2, ..., \mathbf{c}_k] \in \mathbb{R}^{k \times 784} \).

### 4. **RBF Activations**:
For each input \( \mathbf{x}_i \), the activations are computed by applying the RBF function to all \( k \) centers. This generates a new feature vector for each input:
\[
\Phi(\mathbf{x}_i) = [\phi(\mathbf{x}_i, \mathbf{c}_1), \phi(\mathbf{x}_i, \mathbf{c}_2), ..., \phi(\mathbf{x}_i, \mathbf{c}_k)]
\]
This transformation projects the data into a higher-dimensional space, where linear classification is easier.

### 5. **Linear Output Layer**:
After the RBF layer, the network uses a **linear output layer** to classify the transformed data. The weights \( \mathbf{W} \) map the RBF activations \( \Phi(\mathbf{x}_i) \) to class scores for each digit (0–9). The weight matrix \( \mathbf{W} \) has dimensions \( k \times 10 \), where each column corresponds to one class.

For a given input \( \mathbf{x}_i \), the output (class scores) is computed as:
\[
\mathbf{y}_i = \Phi(\mathbf{x}_i) \mathbf{W}
\]
Where \( \mathbf{y}_i \in \mathbb{R}^{10} \) represents the scores for each of the 10 classes. The predicted class is the one with the highest score:
\[
\hat{y}_i = \arg\max(\mathbf{y}_i)
\]

### 6. **Training (Least Squares Method)**:
The output weights \( \mathbf{W} \) are computed using the **least squares solution** to minimize the error between the predicted and true outputs.

You solve the following equation to find \( \mathbf{W} \):
\[
\mathbf{W} = \left( \Phi^T \Phi \right)^{-1} \Phi^T \mathbf{Y}
\]
Where:
- \( \Phi \in \mathbb{R}^{n \times k} \) is the matrix of RBF activations for all \( n \) training samples.
- \( \mathbf{Y} \in \mathbb{R}^{n \times 10} \) is the one-hot encoded matrix of true labels.
  
The least squares method ensures that the linear model fits the transformed data as accurately as possible.

### 7. **Prediction and Evaluation**:
After training, you compute the class scores for the test data using the trained weights \( \mathbf{W} \). The final predicted labels are obtained by selecting the class with the highest score for each sample.

You evaluate the performance using:
- **Accuracy**: The proportion of correctly predicted labels.
- **Mean Squared Error (MSE)**: Measures how close the predicted probabilities are to the true one-hot encoded labels.
  
You also print a **classification report** (precision, recall, F1-score) and plot a **confusion matrix** to analyze how well the model performs on each digit class.

### Summary of Mathematical Mapping:
1. **Input**: \( X \in \mathbb{R}^{n \times 784} \) — Each image is flattened to a vector of 784 pixel values.
2. **K-Means Clustering**: Cluster the training data into \( k \) clusters to get RBF centers \( C \in \mathbb{R}^{k \times 784} \).
3. **RBF Transformation**: Apply RBF functions \( \phi(\mathbf{x}, \mathbf{c}) \) to project the input into a new space \( \Phi(\mathbf{x}) \in \mathbb{R}^{k} \).
4. **Linear Output Layer**: The linear classifier maps the RBF activations to class scores \( \mathbf{y} \in \mathbb{R}^{10} \).
5. **Least Squares Training**: The weights \( \mathbf{W} \) are learned using the least squares method.
6. **Prediction**: Predicted labels are obtained by finding the class with the maximum score \( \hat{y} = \arg\max(\mathbf{y}) \).

This project illustrates how RBF networks can effectively transform input data into a higher-dimensional space, where linear models can achieve high classification performance. In practice, this approach is particularly useful for problems where the decision boundary is nonlinear in the original feature space.