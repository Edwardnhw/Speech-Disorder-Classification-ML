# Speech Disorder Classification Using Machine Learning

**Author**: Hon Wa Ng, @ktolnos, @electronjia, @Fletcher-FJY
**Date**: December 2024  

## Overview

This project classifies speech disorders using machine learning techniques. It extracts acoustic and linguistic features from speech recordings and applies various models, including Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Random Forest, and Deep Learning models (CNN, MLP).

The dataset includes recordings labeled with neurological conditions such as ALS, MSA, PSP, PD, and healthy controls (HC).
## Objectives

- Extract acoustic and linguistic features from speech recordings.
- Train ML models (Logistic Regression, SVM, CNN, MLP) for speech disorder classification.
- Compare classification performance across different models.
- Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.
- Generate spectrograms for feature extraction and augmentation.
- Implement MFCC, jitter, shimmer, intensity, and frequency analysis for feature engineering.

## Dataset
The dataset consists of multiple speech disorder datasets, including:

VOC-ALS Dataset
MINSK Parkinson’s Dataset
MSA Dataset
PD Dataset 1, 2, 3
Italian Parkinson’s Voice Dataset
Each dataset contains:

Raw audio files (.wav format)
Acoustic feature metadata (.csv format)
Labeled speech disorder categories

Dataset Information
This repository contains a cleaned version of the script used and results for speech disorder classification. Due to the large file size, raw datasets are not included in this repository.
If you require the full dataset, please refer to the following repository maintained by my teammate: https://github.com/ktolnos/als-project

## Repository Structure
```bash

speech-disorder-classification-ml/
│── backup-repo/                        # Backup of previous repository
│── doc/
│   │── Final_Project_Report.pdf        # Final project report  
│   │── Project_Proposal.pdf            # Initial proposal document  
│  
│── plots/                              # Model performance visualizations  
│   │── Embeddings_acc_logistic.pdf     
│   │── Embeddings_f1_mlp.png  
│  
│── results/                            # Trained model results and evaluations  
│   │── CNN_results/                    
│  
│── src/                                # Main source code  
│   │── feature_extraction/             # Feature extraction scripts  
│   │   │── extract_acoustic_features.py
│   │   │── generate_spectrograms.py    
│   │── models/                         # Model training scripts  
│   │   │── train_cnn.py                
│   │   │── train_svm_randomforest.py  
│   │── utils/                          # Utility functions  
│  
│── requirements.txt                    # Python dependencies  
│── .gitignore                          # Ignored files  
│── README.md                            # Project documentation  

```

---

## Installation & Usage

### 1. Clone the Repository
```
git clone https://github.com/Edwardnhw/speech-disorder-classification-ml.git
cd speech-disorder-classification-ml

```

### 2. Install Dependencies
```
pip install -r requirements.txt

```

### 3. Add Missing Data Files
Since raw datasets are not included in the repository, ensure that your data/ directory is properly structured:
```
data/
│── raw/                               # Place original datasets here
│── processed/                         # Preprocessed data will be stored here

```

### 4. Run Feature Extraction
To extract acoustic features from the dataset, run:
```
python src/feature_extraction/extract_acoustic_features.py

```
5. Train Machine Learning Models
To train different models, use the following scripts:

Train CNN Model
```
python src/models/train_cnn.py
```
Train Logistic Regression & KNN
```
python src/models/train_logisticRegression_KNN.py
```
Train SVM & Random Forest
```
python src/models/train_svm_randomforest.py
```
6. Evaluate Model Performance
To analyze model results and generate reports:
```
python src/models/evaluate_models.py
```
---
## Methodology
1. Feature Extraction
Extracted acoustic features include:

- Fundamental Frequency (F0) – Mean, median, standard deviation
- Jitter & Shimmer – Measures of voice perturbation
- Intensity – Mean and standard deviation of speech loudness
- MFCCs – Mel-frequency cepstral coefficients for voice characterization
- Spectrogram Analysis – Visual representation of speech signal

2. Data Preprocessing
- Normalization and scaling of extracted features
- Handling missing values through imputation
- Train-test split using stratified sampling
  
3. Machine Learning Models
Implemented models for classification include:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Random Forest
- Convolutional Neural Networks (CNNs)
- Multi-Layer Perceptron (MLP)
  
5. Model Evaluation
Evaluated using:
- Accuracy, Precision, Recall, F1-score
- Confusion Matrices
- Bootstrapped Confidence Intervals


---

## Results Visualization

1. Model Performance on Speech Disorder Classification
CNN & MLP models outperformed traditional ML classifiers on spectrogram-based features.
Logistic Regression and SVM provided good interpretability but lower accuracy.
Feature-based models (MFCC, jitter, shimmer) showed strong correlations with specific disorders.
2. Visualization of Results
Generated plots include:
- Spectrogram visualizations of speech samples
- Confusion matrices for different classifiers
- Accuracy and F1-score heatmaps by vowel and disorder type

---
## Future Improvements
Expand dataset with more diverse speech recordings
Implement Transfer Learning using pre-trained speech models
Enhance data augmentation techniques for improved generalization
Explore waveform-based models using transformers (e.g., Whisper, Wav2Vec 2.0)


---

## Next Steps & Improvements

- Extend analysis to other factors (e.g., programming language, country).
- Implement Machine Learning models for salary prediction.
- Automate data processing with Pandas & NumPy optimizations.

---

## Acknowledgments
Special thanks to:

Dataset providers for sharing valuable speech recordings
OpenAI Whisper & Torchaudio for speech processing tools
Scikit-learn, TensorFlow, PyTorch for machine learning frameworks

---
Final Note
This repository does not include large datasets or raw audio files due to GitHub storage constraints. Please download the data separately and place it in the data/raw/ directory before running the scripts.
---



