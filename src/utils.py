import os
import sys
import logging
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,f1_score
from src.exception import CustomException



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    




def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]

            logging.info(f"Starting training for model: {model_name}")

            gs = GridSearchCV(model, para, cv=3, scoring='accuracy')  
            gs.fit(X_train, y_train)

            logging.info(f"Best parameters for {model_name}: {gs.best_params_}")

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_f1_score = f1_score(y_test, y_test_pred, average='weighted')  

            logging.info(
                f"Metrics for {model_name} - Train Accuracy: {train_accuracy:.4f}, "
                f"Test Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1_score:.4f}"
            )

            report[model_name] = {
                'Train Accuracy': train_accuracy,
                'Test Accuracy': test_accuracy,
                'F1 Score': test_f1_score
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)