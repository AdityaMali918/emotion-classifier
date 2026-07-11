import pickle
import yaml

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier


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
    pickle.dump(clf, open('model.pkl','wb'))

def main():
    X_train, y_train = load_data(r"D:\NEW_PROJECTS\mlops\data\features\train_features.csv")
    params = load_params(params_path="params.yaml")
    model =  training_model(X_train, y_train, params)
    save_model(model)

if __name__ == "__main__":
    main()