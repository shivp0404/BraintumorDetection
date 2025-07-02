# Brain Tumor Detection using Deep Learning

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-Enabled-ff4b4b?logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/TensorFlow-Used-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Accuracy-90%25-brightgreen" alt="Accuracy"/>
  <img src="https://img.shields.io/badge/Python-3.7%2B-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"/>
  <img src="https://img.shields.io/badge/Kaggle-Dataset-blue?logo=kaggle" alt="Kaggle Dataset"/>
</p>

This project is a machine learning web application designed to predict the presence of brain tumors from MRI scans. Built with a Convolutional Neural Network (CNN) model and wrapped in an easy-to-use Streamlit web interface, it allows users to upload MRI images and receive instant predictions.

## Features

- **Web-based Interface:** User-friendly UI built with Streamlit.
- **Deep Learning Model:** Trained on MRI scan images using a CNN architecture.
- **High Accuracy:** Achieves approximately 90% accuracy on the validation set.
- **Multi-class Classification:** Detects four classes—glioma, meningioma, no tumor, and pituitary.
- **Easy Setup:** Clone and run locally with minimal configuration.

## Demo

![Demo Screenshot](/view.png)

## Dataset

- **Source:** [Kaggle - Brain MRI Scans](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Details:** The dataset contains 7023 human brain MRI images classified into four classes: glioma, meningioma, no tumor, and pituitary.

## Libraries Used

- Python (Jupyter Notebook for training)
- Streamlit
- Numpy
- Pandas
- Scikit-learn
- Matplotlib (matplotlib.image, matplotlib.pyplot)
- TensorFlow / Keras (for CNN)

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/shivp0404/brain-tumor.git
cd brain-tumor
```

### 2. Install Dependencies

```bash
pip install streamlit
pip install tensorflow
pip install numpy pandas scikit-learn matplotlib
```

### 3. Run the Web App

```bash
streamlit run app.py
```

## Usage

- Open your browser and go to `http://localhost:8501`
- Upload an MRI image.
- View the prediction and model confidence.

## Model Training Overview

> _These are the main steps followed during training the CNN model:_

1. **Collect the data** from Kaggle (7023 MRI images, 4 classes).
2. **Preprocess the images** (resizing, normalization, etc.).
3. **Split the dataset** into training and validation sets.
4. **Scale the training data** for optimal learning.
5. **Create the model** using Keras Sequential API.
6. **Build the architecture:**
    ```python
    model.add(Conv2D(32, 3, 3, input_shape=(128,128,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_of_classes, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    ```

## Results

- **Accuracy:** ~90% on the validation/test set.

## References

- [Kaggle Brain MRI Scans Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- [Streamlit Documentation](https://docs.streamlit.io/)

## Contact

For any questions or feedback, please contact [shivp0404](https://github.com/shivp0404).
