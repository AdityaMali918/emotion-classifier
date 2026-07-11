import numpy as np
import pandas as pd

import pickle
import json

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score

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
    

    model = load_model(model_path)
    metrics = evaluate(model, test_data_path)
    save_metrics(metrics,save_data_path)

if __name__ == "__main__":
    model_path = r"model.pkl"
    save_data_path = r"metrics.json"
    test_data_path = r'./data/features/test_features.csv'
    main(model_path, save_data_path, test_data_path)    
