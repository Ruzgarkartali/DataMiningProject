import pandas as pd
import functions
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.neural_network import MLPClassifier

#Sur combien d'échantillons voulez-vous ENTRAINER votre machine (max 60 000) :
n_train = 60000

#Sur combien d'échantillons voulez-vous TESTER votre machine :  (max 10 000)"
n_test = 1000

# Chargement des données
df_train = pd.read_csv('./Data/fashion-mnist_train.csv')
df_test = pd.read_csv('./Data/fashion-mnist_test.csv')

taille = []

#on sépare les input et le output
X = df_train.drop(labels = 'label', axis=1).values[0:n_train,:]
y = df_train['label'].values[0:n_train]

#pour vérifier que il y ait à peu près le même nombres de différents cas (ce qui est le cas)
for i in range(len(functions.label_names)):

    itera = np.where(y == i)
    taille.append(np.size(itera))

print("taille : ",taille)


xlab = []
err =[]

c = 256

t = (c,)



for i in range(5,20):

    print(i)
    print("Génération du classifieur en cours ...")


    print("tuple = " , t )
    clf = MLPClassifier(activation = 'logistic',max_iter = 50,verbose = True,hidden_layer_sizes=t,solver ='sgd',learning_rate='adaptive')
    clf.fit(X, y)
    print("Le classifieur a été généré avec succès !", '\n')
    plt.plot(clf.loss_curve_,"ro-")
    plt.show()


    n_error = 0

    print("La machine effectue le test ...")

    for i in range(0, n_test):

        # it = np.random.randint(0, lig)
        test_sample = df_test.drop(labels='label', axis=1).values[i, :]
        answer = df_test['label'].values[i]

        prediction = clf.predict([test_sample])
        prediction = prediction[0]

        # print("     Pour l'image ", it, " la machine a prédit : ", label_names[prediction].upper(), ". Or la bonne réponse est : ", label_names[answer].upper(), '\n')

        if prediction != answer:
            n_error += 1

    print("L'erreur de la machine est de  n_erreurs/n_essais  = ", n_error / n_test * 100, "%", '\n', '\n')

    xlab.append(i)
    err.append(n_error/n_test)
    # = int(c*2)
    #t = t + (c,)


print("err = ",err)

plt.plot(xlab,err,"ro-")
plt.ylabel("Taux d'erreurs")
plt.show()
