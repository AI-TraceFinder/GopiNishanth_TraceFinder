Milestone 3 Deep Learning Model and Explainability

Overview
In this milestone, deep learning techniques were explored to identify the scanner device used to scan a document. The main goal was to study whether a Convolutional Neural Network trained on raw scanned images can learn scanner specific patterns and how its performance compares with classical machine learning methods.

This milestone covers work done during Week 5 and Week 6.

Week 5 CNN Model Development and Training

Objective
To build and train a CNN model using scanned document images and observe its performance under limited data conditions. Image augmentation was also applied to improve generalization.

CNN Trained from Scratch

Approach
A custom Convolutional Neural Network was designed and trained from scratch.
Grayscale scanned document images were used as input.
All images were resized to a fixed resolution for uniform processing.
The dataset contained scanned images from multiple scanner devices, with a limited number of images per scanner.
Basic data augmentation such as small rotations and brightness variation was applied.

Results
Training Accuracy was around 78 percent.
Test Accuracy was around 80+ percent depending on image wise voting.

Observation
The CNN trained from scratch showed underfitting. This means the model was not able to fully learn scanner specific features. This behavior is expected because deep learning models usually require a large amount of data. Scanner identification depends on very subtle noise and texture patterns, which are difficult to learn with limited data.

Week 6 Model Evaluation and Explainability

Objective
To evaluate the CNN model using proper performance metrics and to understand how the model makes decisions using explainability techniques.

Model Evaluation
The model performance was evaluated using accuracy, F1 score, and confusion matrix. Image wise voting was used to combine patch level predictions into final image level predictions. The confusion matrix showed that some scanners were confused with others, indicating similarity in scanner noise patterns.

Explainability using Grad CAM

Purpose
CNN models are often considered black box models. In forensic applications, it is important to ensure that the model focuses on scanner related artifacts rather than document content such as text.

Approach
Grad CAM was applied to visualize the regions of the image that influenced the model’s prediction.

Observations
Grad CAM visualizations showed that the model mainly focused on background areas, paper texture, and noise patterns. Very little attention was given to textual regions. This confirms that the CNN learned scanner specific characteristics rather than semantic information from the document content.

Summary of Milestone 3

The CNN trained from scratch achieved moderate accuracy due to limited dataset size.
Classical machine learning methods performed better on this dataset.
Despite lower accuracy, Grad CAM results validated that the CNN learned meaningful scanner related features.
This milestone demonstrates the strengths and limitations of deep learning for forensic scanner identification under limited data conditions.
