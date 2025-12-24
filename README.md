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
