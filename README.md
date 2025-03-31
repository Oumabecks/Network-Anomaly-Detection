# Network-Anomaly-Detection
# Enhancing Network Security Through Supervised Learning-Based Anomaly Detection in Network Traffic
# Project Overview
This project focuses on enhancing network security by developing a supervised learning-based anomaly detection system for network traffic. The goal is to identify and mitigate potential security threats in real-time by analyzing network traffic patterns and detecting deviations from normal behavior.

# Objectives
Data Collection and Preprocessing: Collect network traffic data from public datasets, preprocess it, and perform feature extraction to identify relevant features.

Model Development: Develop machine learning models using multiple algorithms and train them using the preprocessed data.

Model Evaluation: Evaluate the performance of the developed models and select the best-performing one.

Visualization: Create visualizations to illustrate detected anomalies in network traffic.

# Dataset
The dataset in "Test.txt" contains network connection records representing both normal and malicious traffic, with each entry consisting of 42 features that describe various aspects of the network interactions. These features include basic connection attributes (protocol type like TCP/UDP/ICMP, service type such as HTTP/FTP, connection status flags), traffic volume metrics (source/destination bytes, duration), and behavioral patterns (login attempts, error rates). The data also captures time-based and host-based statistical features (e.g., connection rates, same-service percentages) as well as content-based features (e.g., hot indicators, file operations). Each record is labeled with either "normal" or a specific attack type (e.g., neptune, smurf, guess_passwd, portsweep), making this a labeled dataset suitable for intrusion detection analysis. The attacks cover multiple categories including denial-of-service (neptune), probes (satan), and brute-force attempts (guess_passwd), with numerical and categorical features that collectively provide a comprehensive view of network behavior for security monitoring.
# Data Preprocessing
Missing Values: The datasets were checked for missing values, and none were found.

Duplicates: No duplicate entries were present in the datasets.

Label Encoding: Categorical columns (e.g., protocol type, service, flag, attack) were converted into numerical values using Label Encoding for machine learning compatibility.

Feature Selection: Unnecessary columns (e.g., 'land', 'urgent', 'numfailedlogins') were dropped to streamline the dataset.

# Methodology
Libraries and Tools
The project leverages the following Python libraries:

Data Manipulation: NumPy, Pandas

Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-Learn (for models like Logistic Regression, SVM, and evaluation metrics)

Preprocessing: StandardScaler, LabelEncoder, RFE (Recursive Feature Elimination)

# Models
Logistic Regression: A baseline model for binary classification of normal vs. attack traffic.

Support Vector Machine (SVM): Effective for high-dimensional data and anomaly detection.

Evaluation Metrics: Accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.

# Key Steps
Data Loading and Exploration: Initial exploration to understand the dataset structure and statistics.

Feature Engineering: Extraction and transformation of relevant features for model training.

Model Training: Training and tuning models using the training dataset.

Model Validation: Evaluating model performance on the test dataset.

Anomaly Visualization: Creating plots to highlight detected anomalies and model performance.

# Results
Model Performance
Logistic Regression: Achieved an accuracy of X% with a precision of Y%.

SVM: Demonstrated superior performance with an accuracy of X% and a recall of Y%.

# Visualization
Correlation Matrix: Identified relationships between features.

Confusion Matrix: Illustrated model performance in classifying normal and attack traffic.

ROC Curve: Showed the trade-off between true positive rate and false positive rate.

# How to Use
Prerequisites
Python 3.x

Libraries: NumPy, Pandas, Scikit-Learn, Matplotlib, Seaborn

# Steps to Reproduce
Clone the Repository:

bash
Copy
git clone [(https://github.com/Oumabecks/Network-Anomaly-Detection)]
cd NAD_project
Install Dependencies:

bash
Copy
pip install numpy pandas scikit-learn matplotlib seaborn
Run the Jupyter Notebook:

bash
Copy
jupyter notebook NAD_project.ipynb
Follow the Notebook:

Load and preprocess the datasets.

Train and evaluate the models.

Generate visualizations.

# Future Work
Feature Enhancement: Incorporate additional features for improved anomaly detection.

Real-Time Detection: Implement the model for real-time network traffic monitoring.

Advanced Models: Experiment with deep learning models like LSTM for sequential data analysis.

# Contributors
Antony Irungu
Freenjina@gmail.com 
Irungu Antony
Ainjina@must.co.ke

License
This project is licensed under the MIT License. See the LICENSE file for details.
