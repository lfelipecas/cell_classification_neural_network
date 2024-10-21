# Cell Classification Using Neural Networks

## Project Overview

This project focuses on the **classification of two types of blood cells**: **Basophils** and **Erythroblasts**, using **machine learning** techniques and, specifically, a **neural network model**. The input for the model comes from image data, where features such as **contrast** and **pixel count** are extracted from the images after processing them through **binary thresholding**. These features are then used to train, validate, and test a neural network that can distinguish between the two types of cells.

The project demonstrates the steps required to preprocess the images, extract meaningful features, and implement a **feedforward neural network** for binary classification.

## Project Structure

The project contains the following directories:
```
├── data
│   ├── Data_Cells_Test
│   │   ├── Inference
│   │   │   ├── Basofilos_inference
│   │   │   └── Eritoblastos_inference
│   ├── Data_Cells_Train
│   │   ├── Basofilos_train
│   │   └── Eritroblasto_train
├── LICENSE
├── .gitignore
├── Cell_Classification_Neural_Network_Model.ipynb
├── README.md
└── requirements.txt
```

- **data**: This directory contains the training and test data used to train and evaluate the model. 
  - `Data_Cells_Train`: Contains subfolders for **Basophils** and **Erythroblasts** used for training the model.
  - `Data_Cells_Test`: Contains test data under the `Inference` subdirectory.
  
- **scripts**: Contains the Python scripts that implement the feature extraction, model training, and evaluation processes.
  - `cell_classification.py`: The main script that runs the full pipeline from data preprocessing to model evaluation.
  - `utils.py`: A helper script that contains utility functions for feature extraction and image processing.

## Key Components

### 1. Feature Extraction

We use two key features extracted from each image:
- **Contrast**: Using the Grey Level Co-occurrence Matrix (GLCM) to calculate the texture contrast of the cell.
- **Pixel Count**: The number of white pixels in the binary thresholded image, representing the size of the cell.

These features are extracted using the following functions:
- `apply_binary_threshold`: Converts grayscale images into binary images using a fixed threshold.
- `calculate_contrast`: Computes the contrast using GLCM from the grayscale image.
- `count_pixels`: Counts the number of white pixels in the binary image.

### 2. Neural Network Design

The model is a **feedforward neural network** with the following architecture:
- **Input Layer**: Takes 2 input features (contrast and pixel count).
- **Hidden Layers**: 
  - The first hidden layer has 16 neurons, ReLU activation, and L2 regularization to prevent overfitting.
  - A Dropout layer (0.2) is applied to reduce overfitting.
  - The second hidden layer has 8 neurons with ReLU activation and L2 regularization.
- **Output Layer**: A single neuron with a **sigmoid activation** function to output a probability for binary classification (Basophil or Erythroblast).

### 3. Model Training

The model is trained using the **Adam optimizer** and **binary cross-entropy loss** function, with early stopping applied to prevent overfitting. The model is evaluated on validation and test sets.

Training steps:
1. **StandardScaler** is used to scale the data.
2. The model is compiled with Adam and binary cross-entropy.
3. **Early stopping** is applied during training to stop once validation loss stagnates.
4. The accuracy and loss metrics are plotted to monitor the training process.

### 4. Evaluation

After training, the model is evaluated using:
- **Test Accuracy**: The performance of the model on unseen test data.
- **Confusion Matrix**: Visualizes the model's performance in classifying Basophils and Erythroblasts.

### How to Run the Project

1. Clone the repository:
   ```
   git clone https://github.com/lfelipecas/cell_classification_neural_network
   ```

2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Special thanks to the guidance and dataset provided by Professor Felipe Palta in his lessons on neural networks and machine learning.