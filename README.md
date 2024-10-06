# Spam Filtering Email System using Machine Learning

## Overview
This project focuses on building a **Spam Filtering Email System** using **Logistic Regression**, **Linear Regression**, and a **Perceptron Classifier**. By applying machine learning algorithms, the system classifies emails as spam or non-spam based on specific features extracted from email datasets.

## Features:
- **Data Acquisition:** Utilized publicly available datasets from AWS and Kaggle for spam email classification.
- **Model Development:** Implemented three classifiers:
  - Logistic Regression for binary classification.
  - Linear Regression for continuous data analysis.
  - Perceptron Classifier for a baseline model.
- **Unit Testing:** Thorough unit testing of all classifiers to ensure accurate results.
- **Visualization:** Performance metrics and results visualized in **Jupyter Notebook**.

## Technologies Used:
- **Python** (for model development and testing).
- **Jupyter Notebook** (for environment setup and visualization).
- **Scikit-learn** (for implementing machine learning algorithms).
- **Pandas** and **NumPy** (for data manipulation).
- **Matplotlib** and **Seaborn** (for data visualization).

## How to Run:
1. Clone this repository.
2. Install required libraries using `pip install -r requirements.txt`.
3. Run the Jupyter notebook `spam_filtering.ipynb`.
4. Review unit test results and performance metrics.

## Results:
The Logistic Regression model showed higher accuracy in detecting spam emails compared to Linear Regression and the Perceptron Classifier. The performance metrics for each model are outlined in the project’s notebook.

## Future Improvements:
- Experiment with other classification models like **Random Forest** or **SVM** for better accuracy.
- Explore feature engineering techniques to improve spam detection.
- Implement real-time filtering capabilities in email services.

