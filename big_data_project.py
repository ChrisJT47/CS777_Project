# -*- coding: utf-8 -*-

import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when
import pyspark.sql.functions as F

class COVIDPredictionSystemSpark:
    def __init__(self, spark_session, data_path):

        self.spark = spark_session

        # Read the dataset
        self.df = self.spark.read.csv(
            data_path,
            header=True,
            inferSchema=True
        )

        # Convert categorical columns to numeric
        self.symptom_columns = [
            'Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat',
            'Running Nose', 'Asthma', 'Chronic Lung Disease', 'Headache',
            'Heart Disease', 'Diabetes', 'Hyper Tension', 'Fatigue ',
            'Gastrointestinal ', 'Abroad travel', 'Contact with COVID Patient',
            'Attended Large Gathering', 'Visited Public Exposed Places',
            'Family working in Public Exposed Places', 'Wearing Masks',
            'Sanitization from Market'
        ]

        # Convert 'Yes'/'No' to 1/0
        for col_name in self.symptom_columns + ['COVID-19']:
            self.df = self.df.withColumn(
                col_name,
                when(col(col_name) == 'Yes', 1).otherwise(0)
            )

    def prepare_data(self, test_size=0.2, random_seed=42):

        # Create feature vector
        assembler = VectorAssembler(
            inputCols=self.symptom_columns,
            outputCol="features"
        )

        # Index the target variable
        label_indexer = StringIndexer(
            inputCol="COVID-19",
            outputCol="label"
        )

        # Split the data
        train_data, test_data = self.df.randomSplit(
            [1-test_size, test_size],
            seed=random_seed
        )

        return train_data, test_data, assembler, label_indexer

    def create_model_pipeline(self, assembler, label_indexer):

        # Random Forest Classifier
        rf_classifier = RandomForestClassifier(
            labelCol="label",
            featuresCol="features",
            numTrees=100,
            seed=42
        )

        # Gradient Boosting Classifier
        gbt_classifier = GBTClassifier(
            labelCol="label",
            featuresCol="features",
            maxIter=100,
            seed=42
        )

        # Logistic Regression Classifier
        lr_classifier = LogisticRegression(
            labelCol="label",
            featuresCol="features",
            maxIter=100,
            regParam=0.01,
            elasticNetParam=0.0  # L2 regularization
        )

        # Random Forest Pipeline
        rf_pipeline = Pipeline(stages=[
            label_indexer,
            assembler,
            rf_classifier
        ])

        # Gradient Boosting Pipeline
        gbt_pipeline = Pipeline(stages=[
            label_indexer,
            assembler,
            gbt_classifier
        ])
        
        # Logistic Regression Pipeline
        lr_pipeline = Pipeline(stages=[
            label_indexer,
            assembler,
            lr_classifier
        ])

        return rf_pipeline, gbt_pipeline, lr_pipeline

    def train_and_evaluate_models(self, train_data, test_data, rf_pipeline, gbt_pipeline, lr_pipeline):

        # Train Random Forest Model
        rf_model = rf_pipeline.fit(train_data)
        rf_predictions = rf_model.transform(test_data)

        # Train Gradient Boosting Model
        gbt_model = gbt_pipeline.fit(train_data)
        gbt_predictions = gbt_model.transform(test_data)
        
        # Train Logistic Regression Model
        lr_model = lr_pipeline.fit(train_data)
        lr_predictions = lr_model.transform(test_data)
        

        # Evaluation metrics
        binary_evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            metricName="areaUnderROC"
        )
        multi_evaluator = MulticlassClassificationEvaluator(
            labelCol="label",
            metricName="accuracy"
        )

        # Random Forest Metrics
        print("Random Forest Metrics:")
        print("AUC:", binary_evaluator.evaluate(rf_predictions))
        print("Accuracy:", multi_evaluator.evaluate(rf_predictions))
        
        # Extract and print feature importances
        rf_feature_importances = rf_model.stages[-1].featureImportances
        print("\nFeature Importances for Random Forest:")
        for feature, importance in zip(self.symptom_columns, rf_feature_importances):
            print(f"{feature}: {importance:.4f}")

        # Gradient Boosting Metrics
        print("\nGradient Boosting Metrics:")
        print("AUC:", binary_evaluator.evaluate(gbt_predictions))
        print("Accuracy:", multi_evaluator.evaluate(gbt_predictions))
        
        # Logistic Regression Metrics
        print("\nLogistic Regression Metrics:")
        print("AUC:", binary_evaluator.evaluate(lr_predictions))
        print("Accuracy:", multi_evaluator.evaluate(lr_predictions))

        # Extract and print logistic regression coefficients
        lr_coefficients = lr_model.stages[-1].coefficients
        lr_intercept = lr_model.stages[-1].intercept
        print("\nLogistic Regression Coefficients:")
        for feature, coef in zip(self.symptom_columns, lr_coefficients):
            print(f"{feature}: {coef:.4f}")
            print(f"Intercept: {lr_intercept:.4f}")

        return rf_model, gbt_model, lr_model

    def predict(self, rf_model, gbt_model, lr_model, symptoms):

        # Create a DataFrame from symptoms
        symptoms_df = self.spark.createDataFrame(
            [tuple(symptoms)],
            self.symptom_columns
        )

        # Prepare features for prediction
        symptoms_features = symptoms_df

        # Predict using both models
        rf_result = rf_model.transform(symptoms_features)
        gbt_result = gbt_model.transform(symptoms_features)
        lr_result = lr_model.transform(symptoms_features)
        

        # Collect and process results
        rf_pred = rf_result.select("prediction", "probability").collect()[0]
        gbt_pred = gbt_result.select("prediction", "probability").collect()[0]
        lr_pred = lr_result.select("prediction", "probability").collect()[0]

        return {
            'random_forest_prediction': 'Yes' if rf_pred.prediction == 1 else 'No',
            'random_forest_confidence': float(rf_pred.probability[1]),
            'gradient_boosting_prediction': 'Yes' if gbt_pred.prediction == 1 else 'No',
            'gradient_boosting_confidence': float(gbt_pred.probability[1]),
            'logistic_regression_prediction': 'Yes' if lr_pred.prediction == 1 else 'No',
            'logistic_regression_confidence': float(lr_pred.probability[1])
        }

def main():
    # Create Spark Session
    spark = SparkSession.builder \
        .appName("COVID-19 Prediction System") \
        .getOrCreate()

    # Path to your COVID dataset
    DATA_PATH = sys.argv[1]

    # Initialize the prediction system
    csaps = COVIDPredictionSystemSpark(spark, DATA_PATH)

    # Prepare data
    train_data, test_data, assembler, label_indexer = csaps.prepare_data()

    # Create model pipelines
    rf_pipeline, gbt_pipeline, lr_pipeline = csaps.create_model_pipeline(
        assembler, label_indexer
    )

    # Train and evaluate models
    rf_model, gbt_model, lr_model = csaps.train_and_evaluate_models(
        train_data, test_data, rf_pipeline, gbt_pipeline, lr_pipeline
    )

    # Example predictions from the project proposal
    print("\nExample Predictions:")

    # Input Example 1 from proposal
    example1 = [1,1,1,1,1,0,0,0,1,1,1,1,1,0,1,1,1,1,0,0]
    print("Example 1 Prediction:", csaps.predict(rf_model, gbt_model, lr_model, example1))

    # Input Example 2 from proposal
    example2 = [1,1,1,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0]
    print("Example 2 Prediction:", csaps.predict(rf_model, gbt_model, lr_model, example2))

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    main()
