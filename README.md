# GopiNishanth_TraceFinder
Trace Finder: Forensic Scanner Identification Project Overview The aim of this project is to identify the source scanner device used to scan a document or image by analyzing unique patterns or artifacts left behind during the scanning process. Each scanner brand or model introduces specific noise, texture, or compression traces that can be learned by a machine learning model. This is critical for forensic investigations, document verification, and authentication

Milestone 1: Dataset Collection & Preprocessing In this initial phase, the foundation for the forensic model was established through the following tasks: Data Collection: Collected scanned document samples from multiple scanner devices, including Epson and Canon models

Dataset Labeling: Created a structured dataset with proper labels based on the source device

Image Analysis: Analyzed basic image properties such as resolution, format, and color channels

Preprocessing: Implemented a pipeline to resize all images to a fixed dimension and normalize pixel values for model training

Repository Contents

milestone1_preprocessing.ipynb: Python notebook containing the full preprocessing workflow

dataset_summary.csv: A detailed summary of the scanner models, file counts, and image properties.

Milestone 1 Report.pdf: A formal documentation of the methodology and outcome

Milestone 2 Objectives
Feature Extraction: Implementation of specialized algorithms to extract "micro-features" (LBP, FFT, and Noise Variance).

Dataset Balancing: Managing data distribution across 5 primary scanner models: Canon120-1, Canon200, EpsonV39-1, EpsonV500, and HP.

Model Training: Building a machine learning pipeline using Standard Scaling and Logistic Regression.

Performance Evaluation: Generating a detailed classification report and accuracy metrics.

Technical Implementation
1. Forensic Feature Bank
Rather than training on raw pixels, we engineered a 14-dimensional feature vector for every image:

Local Binary Patterns (LBP0-LBP9): Captures micro-texture patterns and surface variations.

Fast Fourier Transform (FFT Mean/Std): Analyzes the frequency components of the image to detect periodic scanner noise.

Noise Variance (Laplacian): Measures the statistical distribution of high-frequency noise.

2. Model Architecture: Logistic Regression
We implemented a multinomial Logistic Regression model. Key steps included:

Preprocessing: Applied StandardScaler to ensure all features (like LBP and Noise Variance) are on the same mathematical scale.

Optimization: Used the lbfgs solver for efficient multi-class convergence.

Results & Evaluation
The model was tested on a standard 80/20 train-test split, processing 1,348 unique image patches.

Metric	Result
Overall Accuracy	82.22%
Best Performing Class	Canon200 (96% Recall)
Total Samples	1,348 Forensic Patches
Conclusion
The 82.22% accuracy confirms that scanner hardware leaves a distinct, measurable digital fingerprint. While models like the Canon200 are identified with high precision, future milestones could involve deep learning (CNNs) to further improve the classification of more similar models like the Epson series.
Milestone 3 TraceFinder: CNN-based Scanner Identification (Training and Evaluation) Objective

The goal of Milestone 3 is to design, train, evaluate, and explain a deep learning model that can identify the source scanner of a document image based on forensic artifacts.

Key Contributions

Designed a custom convolutional neural network for grayscale document images

Used image-wise dataset splitting to prevent data leakage

Achieved high validation and test accuracy

Performed image-wise voting-based evaluation

Implemented Grad-CAM for model explainability

Dataset Structure dataset/ ├── Cannon120-1/ ├── Cannon200/ ├── EpsonV500/ ├── Epsonv39-1/ └── Hp/

Each folder represents one scanner class

Images are raw scanned document images

Dataset split: 70% training, 15% validation, 15% testing

Splitting is done at image level, not patch level

Model Architecture

Input size: 224 × 224 grayscale images

Four convolutional blocks

Convolution, Batch Normalization, ReLU

Second convolution, Batch Normalization, ReLU

Max pooling and dropout

Global Average Pooling layer

Fully connected classifier (256 → 128 → number of classes)

Total parameters: approximately 1.2 million

Training Configuration

Optimizer: Adam

Loss function: Weighted Cross Entropy Loss

Learning rate scheduler: ReduceLROnPlateau

Early stopping enabled

Training performed on CPU or GPU depending on availability

Evaluation Strategy

Accuracy and loss curves for training and validation

Image-wise evaluation using majority voting

Confusion matrix visualization

Per-class precision, recall, and F1-score

Generated output files:

best_forensic_cnn.pth training_curves.png confusion_matrix.png

Model Explainability (Grad-CAM)

Grad-CAM applied to the final convolutional layer

Highlights scanner-specific artifact regions

Provides forensic interpretability for predictions

Generated output:

gradcam_output.png

Milestone 3 Outcome

Successfully trained a robust CNN model

Achieved strong generalization performance

Added explainability suitable for forensic analysis

Model ready for deployment in Milestone 4
