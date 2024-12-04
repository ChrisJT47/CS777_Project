COVID-19 Symptom Analysis and Prediction System (CSAPS)
BY Tao Jin, Yerbol Baizhumanov, Alsiher Aliyev

Overview

The CSAPS project aims to develop a large-scale analysis system capable of detecting and classifying symptoms to predict whether a patient has COVID-19. It uses three machine learning models: Random Forest, Gradient Boosting Trees, and Logistic Regression. The system is built to scale efficiently for large datasets using Apache Spark. By utilizing cluster computing platforms: Apache Spark's distributed computing capabilities, we aimed to handle large volumes of medical data in a scalable and reliable manner.

Features

1. Data Processing:
- Reads a CSV dataset containing symptom data.
- Converts categorical columns (`Yes/No`) into numeric values (`1/0`).
- Extracts symptoms into a feature vector for model training.
2. Modeling:
- Utilizes three machine learning algorithms:
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Logistic Regression
- Pipelines streamline the process of feature transformation, target indexing, and model fitting.
3. Evaluation:
- Evaluates models using:
  - Binary Classification Metrics: Area Under ROC (AUC)
  - Multiclass Metrics: Accuracy
- Prints feature importances for Random Forest and coefficients for Logistic Regression.
4. Prediction:
- Provides predictions along with confidence scores for all three models.
- Accepts symptom inputs for real-time predictions.
Dependencies
- Use AWS to create cluster or use Google Cloud, use PySpark and Python

Usage

Step 1: Data Preparation
The dataset should be in CSV format with columns representing symptoms (e.g., `Breathing Problem`, `Fever`, etc.) and the target column (`COVID-19`).
Example column headers:
Breathing Problem, Fever, Dry Cough, Sore throat, Running Nose, Asthma, ..., COVID-19
Step 2: Running the Script
Run the script from the command line, passing the path to your dataset as an argument:
python covid_prediction_system.py <path_to_dataset>
Step 3: Example Predictions
The script includes examples for predicting COVID-19 status based on user-provided symptom data. Example input arrays should match the order of symptoms in the dataset.

Code Structure

Class: `COVIDPredictionSystemSpark`
This class implements the end-to-end pipeline for the prediction system.
1. __init__:
- Reads and preprocesses the dataset.
- Converts categorical symptom data to numeric.
2. prepare_data:
- Splits data into training and testing sets.
- Creates a feature vector and indexes the target variable.
3. create_model_pipeline:
- Defines pipelines for Random Forest, Gradient Boosting Trees, and Logistic Regression.
4. train_and_evaluate_models:
- Trains each model.
- Evaluates performance metrics (AUC, accuracy).
- Outputs feature importances and coefficients.
5. predict:
- Provides predictions and confidence scores for new symptom data.

Key Considerations
- The script assumes clean data with consistent column names.
- Model hyperparameters (e.g., tree depth, regularization) can be adjusted for performance tuning.
- Could use neural network implementation to increase efficiency.
