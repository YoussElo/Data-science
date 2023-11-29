### Data Science Portfolio

## Overview
Welcome to my Data Science portfolio! This repository contains a diverse set of projects and exercises related to my Master's studies in Artificial Intelligence. Explore the projects and exercises listed below to get an insight into my skills and interests in the field of data science and machine learning.

## Table of Contents
- Card Fraud Detection
- Twitter Bot Detection
- Sudoku Game Solution using Hill Climbing Algorithm
- Linear Regression on Housing Data
- LDA and QDA on S&P Stock Market Data
- Decision Tree on Bill Authentication Data
- SVM on Credit Card Dataset
- K-Means on Live Dataset
- Technologies Used
- About the Author

# Card Fraud Detection


# Twitter Bot Detection


# Sudoku Game Solution using Hill Climbing Algorithm


# Linear Regression on Housing Data


# LDA and QDA on S&P Stock Market Data
This project explores the utility of Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA) on S&P stock market data. Balanced prior probabilities (0.49 for LDA, 0.51 for QDA) suggest an even distribution of classes in the training data. Group means analysis reveals associations between market directions and preceding days' returns. LDA predicts market trends based on group coefficient magnitudes.

Both LDA and QDA show promise in predicting stock market direction, with LDA exhibiting slightly better performance on this dataset. Further analysis is required to fully comprehend the strengths and limitations of these methods in the context of stock market data, as the overall accuracy of both models remains moderate.

# Decision Tree on Bill Authentication Data
In this project, a Decision Tree Classification approach was employed on the bill_authentication dataset. The code initiates by loading the dataset using pandas, followed by selecting the features and target variables for the model. The features are stored in the variable `X`, while the target is stored in `y`. To ensure standardized features, the StandardScaler class is utilized, transforming them to possess a mean of 0 and a standard deviation of 1.

The data is then divided into training and test sets using the `train_test_split` function, allocating 30% to the test set. A `DecisionTreeClassifier` object is created and fitted to the training data via the `fit` method. Subsequently, the model is evaluated on the test data utilizing the `score` method, and the accuracy is displayed on the console.

The achieved accuracy of 0.99 on the test data indicates that the model accurately classified 99% of the samples, showcasing its robust performance on this dataset. The confusion matrix further reveals detailed performance metrics, illustrating 231 true positives, 1 true negative, 4 false positives, and 176 false negatives. This detailed breakdown provides insights into the model's strengths and areas for potential improvement.

Overall, the project demonstrates the effectiveness of the Decision Tree Classification method in accurately discerning between genuine and counterfeit bills in the bill_authentication dataset.

# SVM on Credit Card Dataset
In this project, the Support Vector Machines (SVM) algorithm was employed to discern between fraudulent and legitimate credit card transactions. The dataset utilized contained historical credit card transactions, and the primary objective was to train the SVM to identify patterns indicative of fraud.

The performance of the SVM model was meticulously assessed using various evaluation metrics, including a confusion matrix, ROC curve, and AUC. The confusion matrix revealed that out of all observations, 56,861 were accurately classified as legitimate transactions, 66 were accurately classified as fraudulent transactions, 32 were erroneously classified as fraudulent transactions, and 3 were mistakenly classified as legitimate transactions.

The ROC curve, illustrating the trade-off between the true positive rate (sensitivity) and false positive rate (1-specificity), along with an AUC value of 0.84, affirmed the model's robust performance. An AUC of 0.84 indicates a good overall performance of the SVM model in distinguishing between fraudulent and legitimate credit card transactions.

In conclusion, the results suggest that the SVM model exhibited a high degree of accuracy in classifying credit card transactions, effectively identifying instances of fraud with confidence.

# K-Means on Live Dataset
In this project, the K-means clustering algorithm was implemented on the "live.csv" dataset. The elbow method was employed to identify the optimal number of clusters. Initially, the analysis suggested 2 clusters as the optimal choice. However, further examination revealed a high inertia value of 237 and a low accuracy of 0.01 for k=2, indicating a suboptimal fit of the model to the data.

Upon revisiting the analysis with k=4, a significant improvement was observed. The model achieved an accuracy of 0.62, indicating a better representation of the underlying patterns in the dataset. Consequently, it is recommended to use 4 clusters for this specific dataset, as it provides a more accurate and meaningful partitioning of the data.

# About the Author


---

