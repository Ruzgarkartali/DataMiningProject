import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import time

label_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]


def loaddata():

    # Chargement des données
    df_train = pd.read_csv('./Data/fashion-mnist_train.csv')
    df_test = pd.read_csv('./Data/fashion-mnist_test.csv')

    return df_train,df_test


def show_img(df,idx):
    x = df.loc[idx]
    target = x.label
    pxs = x[1:].values.reshape(28, 28).astype(float)
    plt.imshow(pxs)
    plt.title(target)
    plt.colorbar()
    plt.axis('off')
    plt.show()


def test(clf,n_test, df_test):

    start = time.time()
    n_error = 0

    ypred = []
    ytrue = []

    print("La machine effectue le test ...")
    #printProgressBar(0, n_test, prefix='Progress:', suffix='Complete', length=50)

    for i in range(0, n_test):

        #pour suivre la progression du test
        if (i%100== 0): print(i)

        test_sample = df_test.drop(labels='label', axis=1).values[i, :]
        answer = df_test['label'].values[i]

        prediction = clf.predict([test_sample])
        prediction = prediction[0]


        ypred.append(prediction)
        ytrue.append(answer)


        #print("     Pour l'image ", i, " la machine a prédit : ", label_names[prediction].upper(), ". Or la bonne réponse est : ", label_names[answer].upper(), '\n')

    #print("L'erreur de la machine est de  n_erreurs/n_essais  = ", n_error / n_test * 100, "%", '\n', '\n')
    end = time.time()
    print("Temps de calcul du test : ", end - start," secondes")

    return ypred, ytrue


def analyse(ypred,ytrue):

    matriceconf = confusion_matrix(ytrue, ypred)

    matriceconf = pd.DataFrame(matriceconf,columns = label_names, index = label_names)



    print("matrice de confusion : ")
    print(matriceconf)

    print("analyse :")
    print(classification_report(ytrue, ypred, target_names=list(label_names)))

    return matriceconf
