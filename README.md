# Handwritten Digits Recognition (PRCP-1002)

A machine learning project that classifies handwritten digit images (0–9) from the **MNIST** dataset and compares classical ML models with a Convolutional Neural Network (CNN).

## Problem Statement

Given a 28×28 grayscale image of a handwritten digit, predict which digit (0–9) it represents, and identify the best-performing classifier.

## Dataset

- **MNIST** — loaded directly with `tensorflow.keras.datasets.mnist` (no manual download needed)
- 60,000 training images and 10,000 test images, each 28×28 pixels (grayscale, values 0–255)
- 10 classes (digits 0–9), roughly balanced (5,421 to 6,742 training images per digit)
- No missing values

## Approach

1. **EDA** — dataset shape, pixel range, class distribution, sample images, single-image pixel matrix
2. **Preprocessing** — normalize pixels to 0–1; flatten images to 784 features for classical models; reshape to 28×28×1 for the CNN
3. **Models trained**
   - Logistic Regression
   - K-Nearest Neighbors (k = 3)
   - Support Vector Machine (RBF kernel)
   - Random Forest (100 trees)
   - CNN (Conv2D → MaxPooling → Dense, Adam optimizer, 5 epochs)
4. **Evaluation** — test accuracy, confusion matrices, classification report
5. **Error analysis** — visualization of misclassified test images

## Results

| Model | Test Accuracy |
|---|---:|
| **CNN** | **98.55%** |
| SVM (RBF) | 97.92% |
| KNN (k=3) | 97.05% |
| Random Forest | 97.04% |
| Logistic Regression | 92.58% |

- The CNN is the best model, misclassifying 145 of 10,000 test images.
- SVM, KNN and Random Forest are all close (about 97–98%), while Logistic Regression is clearly weaker because it can only learn linear boundaries on raw pixels.
- Misclassified digits are mostly ambiguous or unusual handwriting, where different digits look similar.

## Challenges

- Variation in handwriting style, slant and stroke thickness
- Visually similar digits being confused
- Different pixel scales (handled by normalization)
- Choosing among several classifiers (handled by comparing all of them on the same test set)

## Future Improvements

- Deeper CNN with more convolution blocks, dropout and batch normalization
- Data augmentation (small rotations, shifts) for better generalization
- Hyperparameter tuning (k for KNN, C and gamma for SVM, epochs and learning rate for the CNN)
- Compare with more architectures and report results over several random seeds
- Deploy as a small web app where users draw a digit (Streamlit / Flask)

## Tech Stack

Python · TensorFlow / Keras · Scikit-learn · NumPy · Pandas · Matplotlib · Seaborn · Jupyter

## How to Run

```bash
git clone https://github.com/<your-username>/handwritten-digits-recognition.git
cd handwritten-digits-recognition

python -m venv venv
venv\Scripts\activate          # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt

jupyter notebook Handwritten_Digits_Recognition.ipynb
```

MNIST downloads automatically the first time the notebook runs. SVM and KNN can take a few minutes on 60,000 images.

## Project Structure

```
handwritten-digits-recognition/
├── Handwritten_Digits_Recognition.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Author

**Trisha** — AI Engineer Trainee, DataMites
B.Tech CSE · Python · Machine Learning · Deep Learning · Computer Vision

[GitHub](https://github.com/<your-username>) · [LinkedIn](https://linkedin.com/in/<your-handle>)
