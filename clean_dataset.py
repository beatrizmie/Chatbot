import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def cleaner(sentence):
    sentence = sentence.lower()
    sentence = sentence.replace("foxbot", "")
    return sentence

def newSentence(vectorizer, model):
    print("Olá, sou o FoxBot, em que posso ajudar?")
    new = input()

    new_counts = vectorizer.transform([new])
    prediction = model.predict(new_counts)

    print("Você deseja: {0}?".format(prediction[0]))

def main():
    # Lendo o dataset
    df = pd.read_excel('sentencas.xlsx')

    # Limpando o dataset
    df["Sentença"] = df["Sentença"].apply(cleaner)

    # Separando o dataset de treinamento e de teste
    X_train, X_test, y_train, y_test = train_test_split(df["Sentença"], df["Intenção"], test_size=0.33)

    # Classificador Naive-Bayes
    vectorizer = CountVectorizer()
    counts_Xtrain = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(counts_Xtrain, y_train)

    counts_Xtest = vectorizer.transform(X_test)
    y_prob = model.predict(counts_Xtest)

    accuracy = accuracy_score(y_test, y_prob)
    # print("Acurácia do modelo: {:.2f}%".format(accuracy*100))

    print(model.classes_)

    newSentence(vectorizer, model)


if __name__ == "__main__":
    main()