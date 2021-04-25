from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from clean_dataset import clean_dataset

def naive_bayes(X_train, X_test, y_train, y_test):
    # Classificador Naive-Bayes
    vectorizer = CountVectorizer()
    counts_Xtrain = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(counts_Xtrain, y_train)

    counts_Xtest = vectorizer.transform(X_test)
    y_prob = model.predict(counts_Xtest)

    accuracy = accuracy_score(y_test, y_prob)
    # print("Acurácia do modelo: {:.2f}%".format(accuracy*100))

    return vectorizer, model

def classifier():
    X_train, X_test, y_train, y_test = clean_dataset()

    vectorizer, model = naive_bayes(X_train, X_test, y_train, y_test)

    return vectorizer, model

if __name__ == "__main__":
    classifier()