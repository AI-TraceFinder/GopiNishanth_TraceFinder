Milestone 2: Feature Engineering & Classification
Project: TraceFinder Forensic Scanner Identification
Overview
This milestone moves beyond raw image handling to focus on Forensic Feature Extraction and the development of a classification model. We utilize texture analysis and frequency domain transformations to identify the unique "noise" signatures left by different scanner hardwares.

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
🚀 Conclusion
The 82.22% accuracy confirms that scanner hardware leaves a distinct, measurable digital fingerprint. While models like the Canon200 are identified with high precision, future milestones could involve deep learning (CNNs) to further improve the classification of more similar models like the Epson series.
