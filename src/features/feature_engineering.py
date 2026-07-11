import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import logging

from sklearn.feature_extraction.text import TfidfVectorizer

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

def load_params(param_path = "params.yaml"):
    max_features = yaml.safe_load(open(param_path, "r"))['feature_engineering']['max_features']
    return max_features

# fetch the data from processed data
def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    train_data.fillna('',inplace=True)
    test_data.fillna('',inplace=True)
    return train_data, test_data


def feature_engieer(train_data, test_data , max_features):
# appy TFIDF
    
    X_train = train_data['content'].values
    y_train = train_data['sentiment'].values

    X_test = test_data['content'].values
    y_test = test_data['sentiment'].values

    # Apply Bag of Words (CountVectorizer)
    vectorizer = TfidfVectorizer(max_features=max_features)

    # Fit the vectorizer on the training data and transform it
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Transform the test data using the same vectorizer
    X_test_tfidf = vectorizer.transform(X_test)

    train_df = pd.DataFrame(X_train_tfidf.toarray())

    train_df['label'] = y_train

    test_df = pd.DataFrame(X_test_tfidf.toarray())

    test_df['label'] = y_test

    return train_df, test_df

def save_data(train_df, test_df):
    try :


        data_path = Path(".","data", "interim")

        data_path.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(data_path / "train_features.csv", index=False)
        test_df.to_csv(data_path / "test_features.csv", index=False)

    except Exception as e:
         logger.error("Feature data not saved : ", e)
         raise


def main(params_path, train_path, test_path):
       logger.info("----------Feature Extraction Started------------")
       max_features = load_params(params_path)
       train_data, test_data = load_data(train_path, test_path)
       train_df, test_df = feature_engieer(train_data, test_data, max_features=max_features)
       save_data(train_df, test_df)
       logger.info("----------Feature Extraction Finished------------")

if __name__ == "__main__":
       PARAMS_PATH = "params.yaml"
       TRAIN_PATH = "./data/processed/train_processed.csv"
       TEST_PATH = "./data/processed/test_processed.csv"
       main(PARAMS_PATH, TRAIN_PATH, TEST_PATH)
       