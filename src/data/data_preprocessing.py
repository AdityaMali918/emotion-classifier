import numpy as np
import pandas as pd
from pathlib import Path

from src.logger import logger

import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# fetch the data from raw data

def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

# transform the data

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

def lemmatization(text):
    lemmatizer= WordNetLemmatizer()

    text = text.split()

    text=[lemmatizer.lemmatize(y) for y in text]

    return " " .join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):

    text = text.split()

    text=[y.lower() for y in text]

    return " " .join(text)

def removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    try:
        logger.info("--------------------PREPROCESSING------------------")
        df.content=df.content.apply(lambda content : lower_case(content))
        df.content=df.content.apply(lambda content : remove_stop_words(content))
        df.content=df.content.apply(lambda content : removing_numbers(content))
        df.content=df.content.apply(lambda content : removing_punctuations(content))
        df.content=df.content.apply(lambda content : removing_urls(content))
        df.content=df.content.apply(lambda content : lemmatization(content))
        logger.info("--------------------PREPROCESSING DONE------------------")
        return df
    except Exception as e:
        logger.error("Error occured : ",e)
        raise


def save_data(train_processed_data, test_processed_data):
    try:
        file_path = Path(__file__).parent.parent

        data_path = Path(file_path,"data", "processed")

        data_path.mkdir(parents=True, exist_ok=True)

        train_processed_data.to_csv(data_path.joinpath("train_processed.csv"), index=False)
        test_processed_data.to_csv(data_path.joinpath("test_processed.csv"), index=False)
    except Exception as e:
        logger.error("Error occured while storing the processed file ", e)
        raise    
    

def main(train_path, test_path):

    train_data, test_data = load_data(train_path, test_path)

    train_processed_data = normalize_text(train_data)
    test_processed_data = normalize_text(test_data)

    save_data(train_processed_data, test_processed_data)
# store the data 

if __name__ == "__main__":
    TRAIN_PATH = "./data/raw/train.csv"
    TEST_PATH = "./data/raw/test.csv"
    main(TRAIN_PATH,TEST_PATH)

