import pickle
import yaml
import logging

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

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

def load_params(params_path = "params.yaml"):
    params = yaml.safe_load(open(params_path, "r"))['model_building']
    return params


# fetch the data from data/processed
def load_data(train_data_path):
    train_data = pd.read_csv(train_data_path)

    X_train = train_data.iloc[:,0:-1].values
    y_train = train_data.iloc[:,-1].values

    return X_train, y_train

# Define and train the XGBoost model
def training_model(X_train, y_train, params):
    clf = GradientBoostingClassifier(n_estimators=params['n_estimators'], learning_rate= params['learning_rate'])
    clf.fit(X_train, y_train)
    return clf

# save
def save_model(clf):
    pickle.dump(clf, open('.\model\model.pkl','wb'))

def main():
    try:
        logger.info("Training started")
        X_train, y_train = load_data(r".\data\interim\train_features.csv")
        params = load_params(params_path="params.yaml")
        model =  training_model(X_train, y_train, params)
        save_model(model)
        logger.info("Training completed")
    except Exception as e:
        logger.error("Error while training: ",e)  
        raise  

if __name__ == "__main__":
    main()