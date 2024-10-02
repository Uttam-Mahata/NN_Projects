Mapping the **Self-Organizing Map (SOM)** implementation to its underlying **mathematical theory** involves understanding the core principles and how they relate to both the algorithm and its practical applications. Here's how you can relate the **mathematical problem** to the SOM theory, using the image compression project as an example.

### **1. Overview of SOM Theory**

A **Self-Organizing Map (SOM)** is an unsupervised neural network used to map high-dimensional data onto a lower-dimensional (often 2D) grid. The goal is to preserve the topological structure of the data, meaning that similar inputs should map to neighboring nodes.

### **2. Mathematical Formulation of SOM**

Let's break down the key mathematical components of SOM:

#### **Step 1: Input Data Representation**
- **Input vector** $$ \mathbf{x} \in \mathbb{R}^n $$: Each input to the SOM is a vector of $$ n $$ dimensions (in this case, the flattened image pixel values).
  - For CIFAR-10, $$ n = 32 \times 32 \times 3 = 3072 $$.
  - Each image pixel is flattened into a 1D vector $$ \mathbf{x} = (x_1, x_2, ..., x_{3072}) $$.

#### **Step 2: SOM Grid and Weights**
- **Neuron weight vector** $$ \mathbf{w}_i \in \mathbb{R}^n $$: Each neuron (or node) on the SOM grid has a corresponding weight vector of the same dimension as the input vector $$ \mathbf{x} $$.
  - For a 20x20 SOM grid, there are 400 neurons, each with a weight vector $$ \mathbf{w}_i \in \mathbb{R}^{3072} $$.

#### **Step 3: Best Matching Unit (BMU)**
- **Distance Function**: To find the best matching unit (BMU) for each input vector $$ \mathbf{x} $$, the algorithm computes the Euclidean distance between the input vector and each neuron's weight vector:
  $$
  d(\mathbf{x}, \mathbf{w}_i) = \|\mathbf{x} - \mathbf{w}_i\| = \sqrt{\sum_{j=1}^{n} (x_j - w_{ij})^2}
  $$
  - The neuron with the smallest distance to the input vector $$ \mathbf{x} $$ becomes the **BMU**.
  - Mathematically, the BMU $$ \mathbf{w}_{\text{BMU}} $$ is determined by:
  $$
  \mathbf{w}_{\text{BMU}} = \arg\min_{i} \|\mathbf{x} - \mathbf{w}_i\|
  $$

#### **Step 4: Weight Update (Learning Rule)**
- Once the BMU is identified, the weights of the BMU and its neighboring neurons are updated. This process is governed by **Kohonen's learning rule**, which moves the weight vectors closer to the input vector $$ \mathbf{x} $$ over time:
  $$
  \mathbf{w}_i(t+1) = \mathbf{w}_i(t) + \alpha(t) \cdot h_{i,\text{BMU}}(t) \cdot (\mathbf{x} - \mathbf{w}_i(t))
  $$
  - $$ \alpha(t) $$: Learning rate at time $$ t $$, which typically decays over time.
  - $$ h_{i,\text{BMU}}(t) $$: Neighborhood function, which defines how much neighboring neurons should be updated. Typically, this is a Gaussian function:
    $$
    h_{i,\text{BMU}}(t) = \exp\left(-\frac{d_{\text{grid}}(i, \text{BMU})^2}{2\sigma(t)^2}\right)
    $$
    where $$ d_{\text{grid}}(i, \text{BMU}) $$ is the distance between the BMU and neuron $$ i $$ on the SOM grid, and $$ \sigma(t) $$ is the neighborhood radius, which also decays over time.

### **3. Mapping SOM to the Image Compression Problem**

The **image compression problem** can be viewed as a **vector quantization problem** where the input data (pixels) are quantized to a smaller set of representative colors. Here's how this works mathematically:

#### **Step 1: Input Image Representation**
- Each **pixel** in the image is treated as a **vector** $$ \mathbf{x} \in \mathbb{R}^3 $$, where the three dimensions represent the RGB values.
  - For the CIFAR-10 images, $$ \mathbf{x} = (R, G, B) $$, with values normalized between 0 and 1.

#### **Step 2: Mapping to SOM**
- For each pixel, the **BMU** on the SOM grid is identified by minimizing the Euclidean distance between the pixel value and the neuron weight vector.
  - The BMU serves as the **quantized color** for that pixel, effectively compressing the image.

#### **Step 3: Compression**
- Instead of storing the entire 32x32 pixel RGB values (which require 3072 values), the compressed image is represented by the **BMU indices** for each pixel. This reduces the color space from $$ 255^3 $$ possible colors to the number of neurons in the SOM grid.
  - In the case of a 20x20 SOM grid, the number of possible colors is reduced to 400.

#### **Step 4: Reconstruction**
- The compressed image is reconstructed by mapping each pixel to the RGB value of its corresponding BMU's weight vector.

#### **Step 5: Error Measurement (MSE, SSIM)**
- The **Mean Squared Error (MSE)** and **Structural Similarity Index (SSIM)** are computed to quantify the quality of the compression.
  - **MSE**: Measures the average squared difference between the original and compressed images:
    $$
    \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{x}_i)^2
    $$
    where $$ x_i $$ is the original pixel value and $$ \hat{x}_i $$ is the compressed pixel value.
  - **SSIM**: Measures the perceptual similarity between the original and compressed images, considering luminance, contrast, and structure.

### **4. Mathematical Summary**

To summarize the **mathematical problem** that the SOM solves in the context of image compression:
- **Input space**: Each pixel in the image is a vector in $$ \mathbb{R}^3 $$ (RGB values).
- **Quantization**: The SOM maps each input pixel to the nearest neuron in the grid (the BMU).
- **Learning**: The SOM iteratively adjusts the weights of the neurons so that they better represent the input data (i.e., the colors in the image).
- **Compression**: By replacing each pixel with its corresponding BMU, the image is compressed into a lower-dimensional space, represented by fewer colors.
- **Reconstruction**: The compressed image is reconstructed by replacing each pixel with the color of its BMU.

### **5. Extension to the Neural Network Project**
- **Dynamic Learning Rate**: The learning rate $$ \alpha(t) $$ decays over time, improving convergence and ensuring that the SOM stabilizes as training progresses.
- **Neighborhood Function**: The neighborhood size $$ \sigma(t) $$ also decays over time, starting with large updates to the entire SOM and then fine-tuning specific neurons.

### **6. Visualization of SOM Grid as a Reduced Color Palette**
- Once the SOM is trained, the neuron weights represent the reduced color palette. Each neuron’s weight vector corresponds to a color in the image, and you can visualize this as a 20x20 grid of colors.

### **7. Mathematical Benefits of Using SOM for Image Compression**
- **Topological Preservation**: The SOM retains the relative distance between similar colors, ensuring smooth transitions in compressed images.
- **Dimensionality Reduction**: The SOM reduces the dimensionality of the color space, which is beneficial for storage and processing in applications like image compression.

---

By framing the image compression task as a vector quantization problem and using the SOM to quantize the colors in the image, you can mathematically describe the problem and solution. The algorithm minimizes the quantization error (i.e., the difference between the original and compressed pixel values) while preserving the topological relationships between colors.