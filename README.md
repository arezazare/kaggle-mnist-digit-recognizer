# 🔢 MNIST Digit Recognizer – Kaggle Competition

## 📌 Overview
This project tackles the classic MNIST digit classification challenge hosted on [Kaggle](https://www.kaggle.com/c/digit-recognizer). The goal is to identify handwritten digits (0–9) based on pixel values from grayscale images.

Two full modeling pipelines were implemented:
- ✅ A classical machine learning approach using Scikit-learn
- ✅ A deep learning approach using Convolutional Neural Networks (CNNs) built with Keras

> 🎯 **Final Kaggle Scores**:
- **Classical ML Models**: 96.86% accuracy
- **CNN Model**: **98.84% accuracy**

---

## 📊 Dataset
- **Source**: [Kaggle Digit Recognizer Dataset](https://www.kaggle.com/c/digit-recognizer/data)
- **Training Set**: 42,000 images (28×28 grayscale pixels)
- **Test Set**: 28,000 images (unlabeled)
- **Features**: 784 pixel values (`pixel0` to `pixel783`)
- **Target**: `label` (digit from 0 to 9)

---

## 🧪 Key Tasks
- Data exploration and visualization
- PCA for visualizing digit separation
- Preprocessing and normalization
- Feature scaling for classical models
- Training with multiple ML algorithms
- Designing, compiling, and training a CNN model
- Implementing callbacks: EarlyStopping, ReduceLROnPlateau
- Using image augmentation to improve generalization
- Model evaluation with confusion matrix, F1 score, and classification report
- Final prediction and Kaggle submission

---

## 🧠 Models Used

### 🔎 Classical ML Models (Scikit-learn & XGBoost)
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- XGBoost

> 🏁 **Best Classical ML Score**: 0.9686 (XGBoost)

### 🧠 Deep Learning Models (Keras + TensorFlow)
- Convolutional Neural Network with:
  - 3 × Conv2D + MaxPooling2D
  - BatchNormalization & Dropout
  - Softmax output layer
  - Data Augmentation with `ImageDataGenerator`
  - EarlyStopping + ReduceLROnPlateau callbacks

> 🏆 **Best CNN Score**: 0.9884

---

## 📈 Results

| Model Type | Score |
|------------|-------|
| XGBoost (ML) | 0.9686 |
| CNN (Keras)  | **0.9884** ✅ |

- CNN performance surpassed all traditional models
- Ensemble & tuning strategies were explored to improve accuracy
- Training/Validation curves show strong generalization with minimal overfitting

---

## 🔧 Technologies Used
- Python (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost)
- TensorFlow / Keras
- Jupyter Notebook
- Kaggle Kernels & APIs
- Git & GitHub

---

## 🧠 What I Learned
- Hands-on application of CNN architecture for image classification
- Performance differences between ML vs DL in image tasks
- The impact of tuning, augmentation, and callbacks on model performance
- How to transition from EDA to modeling to Kaggle-ready submission

---

## 👏 Author
**Reza Zare** – Full pipeline design, modeling, evaluation, and deployment.

- 📧 [ahmad.r.zarre@gmail.com](mailto:ahmad.r.zarre@gmail.com)  
- 🌐 [LinkedIn](https://www.linkedin.com/in/arezazare)  
- 🧠 [Portfolio Website](https://arezazare.github.io)

---

## 📎 Credits
Thanks to [Kaggle](https://www.kaggle.com/c/digit-recognizer) and the MNIST dataset creators. This project was completed for educational and portfolio purposes.
