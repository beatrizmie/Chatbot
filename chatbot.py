from classifier import classifier

def newSentence(vectorizer, model):
    print("Como posso ajudar? (digite 'sair' para encerrar comunicação)")
    new = input()

    if new == "sair":
        print(" ")
        print("Espero ter ajudado, até logo!")
        return False

    new_counts = vectorizer.transform([new])
    prediction = model.predict(new_counts)

    if prediction[0] == "Não sei":
        resposta = "s"
        print("Desculpe, não entendi")
    else:
        resposta = input("Você deseja: {0}? [s/n] ".format(prediction[0]))
        if resposta == "s":
            if prediction[0] == "Consultar saldo da conta":
                print("Você possui R$ 13.570,00 na conta.")
            elif prediction[0] == "Obter informações relativas ao clima":
                print("A temperatura neste momento é de 23°, com uma pequena chance de chuva.")
            else:
                print("Status da luz: ligada. \nStatus do ar condicionado: desligado.")

    prediction_list = [prediction[0]]

    while resposta == "n":
        for i in model.classes_:
            if i not in prediction_list:
                if i == "Não sei":
                    print("Desculpe, não entendi")
                    resposta = "s"
                else:
                    resposta = input("Então você deseja: {0}? [s/n] ".format(i))
                    prediction_list.append(i)
                    if resposta == "s":
                        if i == "Consultar saldo da conta":
                            print("Você possui R$ 13.570,00 na conta.")
                        elif i == "Obter informações relativas ao clima":
                            print("A temperatura neste momento é de 23°, com uma pequena chance de chuva.")
                        else:
                            print("Status da luz: ligada. \nStatus do ar condicionado: desligado.")
                break
        if len(prediction_list) == len(model.classes_):
            print("Desculpe, não consigo te ajudar. :(")
            break

    print(" ")
    return True

def chatbot():
    vectorizer, model = classifier()

    print("Olá, eu sou o FoxBot!")
    print("---------------------")
    print(" ")

    roda = True
    while roda:
        roda = newSentence(vectorizer, model)

if __name__ == "__main__":
    chatbot()