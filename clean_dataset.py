import pandas as pd
import re
from sklearn.model_selection import train_test_split

def cleaner(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"[+-]", "", sentence)
    sentence = sentence.replace("foxbot", "")
    return sentence

def clean_dataset():
    # Lendo o dataset
    df = pd.read_excel('sentencas.xlsx')

    # Limpando o dataset
    df["Sentença"] = df["Sentença"].apply(cleaner)

    # Separando o dataset de treinamento e de teste
    X_train, X_test, y_train, y_test = train_test_split(df["Sentença"], df["Intenção"], test_size=0.33, random_state=666, stratify=df["Intenção"])

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    clean_dataset()