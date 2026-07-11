import yaml

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

from src.logger import logger

def load_params(params_path = "params.yaml") -> pd.DataFrame:
    try:
        logger.info(f"Params loaded")
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        return params["data_ingestion"]["test_size"]
    except FileNotFoundError:
        logger.error(f"Error: '{params_path}' not found.")
        raise


def load_data(url):
    try : 
        logger.info(f"Loading dataset from {url}")
        df = pd.read_csv(url)
        return df
    except Exception as e:
        logger.error(f"Message : {e}")
        raise


def preprocess_data(df):

    logger.info("Starting preprocessing")

    df.drop(columns=['tweet_id'],inplace=True)

    final_df = df[df['sentiment'].isin(['happiness','sadness'])]

    final_df.replace({
        'sentiment': {
            'happiness': 1,
            'sadness': 0
        }
    }, inplace=True)

    logger.info("Preprocessing completed. Shape: %s", final_df.shape)

    return final_df


def save_data( train_data, test_data):
    try:
        file_path = Path(__file__).parent.parent
        data_path = file_path / "data" / "raw"

        data_path.mkdir(parents=True, exist_ok=True)

        train_data.to_csv(data_path / "train.csv", index=False)
        test_data.to_csv(data_path / "test.csv", index=False)

        print("Train and test data saved successfully.")

    except Exception as e:
        logger.exception("Error while saving data")
        raise


def main(url, params_path):

    logger.info("========== Data Ingestion Started ==========")

    df = load_data(url)

    test_size = load_params(params_path)

    processed_df = preprocess_data(df)

    train_data, test_data = train_test_split(processed_df, test_size=test_size, random_state=42) 

    save_data(train_data, test_data)

    logger.info(
        "Train Shape: %s | Test Shape: %s",
        train_data.shape,
        test_data.shape,
    )

    logger.info("========== Data Ingestion Completed ==========")

if __name__ == "__main__":

    URL = "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"
    PARAMS_PATH = "params.yaml"
    main(URL, PARAMS_PATH)



