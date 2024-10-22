# Cell Classification Neural Network Model

## 1. Introduction

This project is a binary classification model designed to classify two types of cells: **Basophils** and **Erythroblasts**. The project utilizes a simple deep learning model trained on features extracted from the grayscale images of these cells. The neural network model is implemented using **TensorFlow** and **Keras**, while image processing is handled with **OpenCV** and **Scikit-image**.

The workflow consists of:
- Image preprocessing
- Feature extraction using Gray Level Co-occurrence Matrix (GLCM)
- Data visualization and exploration
- Neural network model design and training
- Model evaluation with accuracy, loss, and confusion matrix plots

The dataset is divided into **training**, **validation**, and **test** sets, and a binary classification model is trained on two features: **Contrast** and **Pixel Count** extracted from the processed images.

## 2. Project Structure

The project structure is as follows:

```
├── data
│   ├── Data_Cells_Train
│   │   ├── Basofilos_train
│   │   └── Eritroblasto_train
│   ├── Data_Cells_Test
│   │   ├── Inference
│   │   │   ├── Basofilos_inference
│   │   │   └── Eritroblastos_inference
├── Cell_Classification_Model.ipynb
├── README.md
├── .gitignore
└── LICENSE
```

- `data/` contains the images for training and inference.
- `Cell_Classification_Model.ipynb` contains the code implementation.
- `.gitignore` ensures that unnecessary files (e.g., compiled Python files, virtual environments) are not included in the repository.
- `LICENSE` includes the licensing information.

## 3. Dependencies

The project requires the following libraries:
- `numpy`
- `opencv-python`
- `matplotlib`
- `pandas`
- `tensorflow`
- `seaborn`
- `scikit-image`
- `scikit-learn`

Install the required packages using the following command:

```
pip install -r requirements.txt
```

## 4. Image Processing Functions

The project includes several image processing functions:
- **Binary Thresholding**: Applies an inverted binary threshold to convert grayscale images to binary images.
- **Contrast Calculation**: Uses GLCM (Gray Level Co-occurrence Matrix) to calculate the contrast of the image.
- **Pixel Count**: Counts the number of white pixels (value 255) in the binary image.

Each image is processed to extract these features, which are used as inputs to the neural network model.

## 5. Data Visualization

### 5.1 Processed Images Visualization

The function `visualize_processed_images_by_rows()` is used to visualize a set of randomly selected processed images from each category. The processed images (binary images) are displayed in two rows:
- The first row contains **Basophil** images.
- The second row contains **Erythroblast** images.

Example:

```
random_basophil_images = select_random_images(basofilos_train_dir, num_images=10)
random_erythroblast_images = select_random_images(eritroblasto_train_dir, num_images=10)
visualize_processed_images_by_rows(random_basophil_images, random_erythroblast_images, num_images=10)
```

### 5.2 DataFrame and Scatter Plot

A `DataFrame` is created using the extracted features (Contrast, Pixel Count) and their respective labels (Basophil, Erythroblast). A scatter plot is then generated to show the distribution of **Contrast** vs **Pixel Count** for both classes.

```
df_train = create_dataframe(X_train_scaled, y_train, columns=["Contrast", "Pixel Count"])
plot_scatter(df_train)
```

## 6. Neural Network Model

The neural network used in this project is a **feed-forward fully connected network**. It is composed of:
- **Input layer**: Accepts two features (Contrast and Pixel Count).
- **Hidden layers**: Two dense layers with `ReLU` activation, each followed by **L2 regularization** to prevent overfitting.
- **Dropout layer**: Added to randomly deactivate 20% of the neurons during training to avoid overfitting.
- **Output layer**: A single neuron with **sigmoid activation** for binary classification.

The model is compiled with the **Adam optimizer**, using **binary crossentropy** as the loss function.

The neural network architecture is defined as follows:

```
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),  
    tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)),  
    Dropout(0.2, seed=seed),
    tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.001)),  
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### 6.1 Model Training

The model is trained on the training data with early stopping to prevent overfitting. The training process tracks the accuracy and loss over epochs.

```
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train_scaled, y_train, epochs=100, validation_data=(X_val_scaled, y_val), batch_size=8, callbacks=[early_stopping])
```

## 7. Performance Evaluation

After training, the model's performance is evaluated on the test set, and several plots are generated:
- **Accuracy and Loss Curves**: Plots showing the training and validation accuracy and loss over epochs.
- **Confusion Matrix**: A matrix showing the number of correct and incorrect predictions for each class.

```
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Basophil", "Erythroblast"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
```

## 8. Results

The final model is evaluated on the test set, and performance metrics such as **accuracy** and **loss** are calculated. The confusion matrix provides insights into the number of correctly classified cells from both classes.

The trained model achieves:

- **Test Loss**: The final loss value on the test set.
- **Test Accuracy**: The final accuracy of the model on the test set.

## 9. Conclusion

This project demonstrates a simple and effective approach to binary classification of cell images using basic image processing techniques and a neural network model. By extracting key features (Contrast and Pixel Count), we are able to classify Basophils and Erythroblasts with high accuracy. The project also emphasizes good practices such as the use of regularization and dropout to prevent overfitting and the visualization of results for better interpretability.

## 10. License

This project is licensed under the MIT License - see the LICENSE file for details.

## 11. Acknowledgments
Special thanks to the guidance and dataset provided by Professor Felipe Palta in his lessons on neural networks and machine learning.