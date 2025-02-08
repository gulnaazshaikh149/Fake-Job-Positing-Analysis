# Fake-Job-Positing-Analysis

# Predictive Modeling for Fake Job Postings using Machine Learning Classifiers
## Overview
This project aims to detect fraudulent job postings by building predictive models using machine learning classifiers. The dataset used contains over 17,000 rows and 20+ features representing various characteristics of job postings. Through data preprocessing, feature engineering, and the application of Stochastic Gradient Descent (SGD) Classifier along with other models, we successfully achieved an F1 score of 85% for distinguishing between legitimate and fraudulent job postings.

## Features
- Data Preprocessing: Handling missing values, removing duplicates, and encoding categorical variables
- Class Imbalance Handling: Implemented techniques to manage class imbalance for accurate predictions
- Machine Learning Models: Tested multiple classifiers, including Stochastic Gradient Descent, Logistic Regression, Random Forest, and XGBoost
- Evaluation Metrics: Focused on F1 Score due to the importance of balancing precision and recall for fraud detection

## Dataset
The dataset contains over 17,000 job postings with 20+ features such as job title, location, salary range, employment type, and job description. The target variable indicates whether a job posting is fraudulent or not.

## Workflow
### Data Preprocessing
-Removed duplicates and irrelevant columns
-Handled missing values
-Applied feature scaling and encoding for categorical variables

### Model Selection and Training
- Implemented Stochastic Gradient Descent (SGD) Classifier
- Tested and compared performance with Logistic Regression, Random Forest, and XGBoost
- Used SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance

### Model Evaluation
- Achieved an F1 Score of 85%, ensuring a balance between precision and recall
- Compared evaluation metrics (accuracy, precision, recall) across all models

## Results
**Best Model:** Stochastic Gradient Descent Classifier
**F1 Score:** 85%
Improved detection of fraudulent job postings while minimizing false positives

## Technologies Used
Python (NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn)
Jupyter Notebook

## Future Work
- Implement deep learning models for better accuracy
- Deploy the best-performing model using Flask or Django
- Perform real-time predictions with streaming data
