import numpy as np
import pandas as pd

import pickle
import json
import logging

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(r"D:\NEW_PROJECTS\mlops\emotion-classifier\logger\logs\running.log", mode="a")
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def load_model(model_path: str):
    clf = pickle.load(open(model_path,'rb'))
    return clf

def evaluate(clf, test_data_path : str) -> dict:  
    test_data = pd.read_csv(test_data_path)

    X_test = test_data.iloc[:,0:-1].values
    y_test = test_data.iloc[:,-1].values

    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    metrics_dict={
        'accuracy':accuracy,
        'precision':precision,
        'recall':recall,
        'auc':auc
    }

    return metrics_dict

def save_metrics(metrics_dict, save_data_path):
    with open(save_data_path, 'w') as file:
        json.dump(metrics_dict, file, indent=4)

def main(model_path, save_data_path, test_data_path):
    try : 
        logging.info("Evaluation started")
        model = load_model(model_path)
        metrics = evaluate(model, test_data_path)
        save_metrics(metrics,save_data_path)
        logging.info("Evaluation Completed")
    except Exception as e:
        logger.error("Eroor occured while evaluation:",e)    
        raise

if __name__ == "__main__":
    model_path = r"./model/model.pkl"
    save_data_path = r"reports/metrics.json"
    test_data_path = r'./data/interim/test_features.csv'
    main(model_path, save_data_path, test_data_path)    
