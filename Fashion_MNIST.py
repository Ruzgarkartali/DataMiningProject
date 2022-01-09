import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import functions
import time
from sklearn.neighbors import KNeighborsClassifier
import numpy as np





print("Sur combien d'échantillons voulez-vous ENTRAINER votre machine (max 60 000) :")
n_train = input()
n_train = int(n_train)

print("Sur combien d'échantillons voulez-vous TESTER votre machine :  (max 10 000)")
n_test = input()
n_test = int(n_test)



#Chargement des données
df_train = pd.read_csv('./Data/fashion-mnist_train.csv')
df_test = pd.read_csv('./Data/fashion-mnist_test.csv')


#on sépare les input et le output
X = df_train.drop(labels = 'label', axis=1).values[0:n_train,:]
y = df_train['label'].values[0:n_train]

X_test = df_test.drop(labels='label', axis=1).values[:, :]


choix = 0

while(choix < 5):

    print(" ********* MENU **************")
    print("1) random forest")
    print("2) svm ")
    print("3) réseaux de neurones")
    print("4) K neighbors")
    print('\n')
    print("Entrez un choix : ")
    choix = input()
    choix = int(choix)

    if(choix == 1):
        print('\n')
        print("***************** RANDOM FORESTS ***************************",'\n')

        print("Génération du classifieur en cours ...")
        start = time.time()

        clf = RandomForestClassifier()
        clf.fit(X, y)
        print("Le classifieur a été généré avec succès !",'\n')

        end = time.time()
        print("Temps de calcul de la génération : ", end - start, " secondes")

        ypred, ytrue = functions.test(clf,n_test,df_test)
        print("ypred = ", ypred)
        print("ytrue = ", ytrue)
        M = functions.analyse(ypred, ytrue)
        M.to_csv("Matrice de confusion RF.csv")



        print('\n')


    if(choix == 2):


        print("***************** SVM ***************************",'\n')

        print("voulez-vous appliquez l'acp ?(oui = y | non  = n)")
        c = input()

        if c == 'y':
            a = np.concatenate((X, X_test), axis=0)

            pca = PCA(.95)

            pca.fit(a)
            a = pca.transform(a)

            X = a[:60000][:]
            X_test = a[60000:70000][:]





        print("Génération du classifieur en cours ...")
        start = time.time()

        clf = svm.SVC(kernel = 'rbf', C= 1000)
        clf.fit(X, y)
        print("Le classifieur a été généré avec succès !",'\n')
        end = time.time()
        print("Temps de calcul de la génération : ", end - start, " secondes")

        #ypred, ytrue = functions.test(clf,n_test,df_test)
        ypred = clf.predict(X_test)
        ytrue = df_test['label'].values[0:n_test]

        M = functions.analyse(ypred, ytrue)
        M.to_csv("Matrice de confusion SVM.csv")

        print('\n')


    if(choix == 3):
        print("***************** RESEAUX DE NEURONES ***************************",'\n')

        print("Génération du classifieur en cours ...")
        start = time.time()
        #améliorer le résultat,: nbre de couches cachées(512, 128,,...), la fonction d'activation(log,tanh,relu), solver(stochastic gradoent des, quasinewton, adam)
        #changer alpha(= la pénalité de la fonction l2), learning_rate = constant ou sacling inversé soit adaptive(maxime mettrait ça) sert seulement si on utilise gradient descent
        clf = MLPClassifier(activation = 'logistic',max_iter = 30,verbose = True,hidden_layer_sizes=(256,),solver = 'sgd', learning_rate='adaptive')


        clf.fit(X, y)
        #print("maxime : ", "/ ", clf.n_outputs_)
       #print("maxime2 : ",clf.n_layers_,"  / ",clf.n_features_in_,"  /   / ",clf.loss_curve_)

        print("Le classifieur a été généré avec succès !",'\n')
        end = time.time()
        print("Temps de calcul de la génération : ", end - start, " secondes")

        ypred, ytrue = functions.test(clf,n_test,df_test)
        M = functions.analyse(ypred, ytrue)
        M.to_csv("Matrice de confusion NN.csv")

        print('\n')

    if(choix == 4):
        print("***************** K-NEAREST NEIGHBORS ***************************", '\n')

        print("Génération du classifieur en cours ...")
        start = time.time()
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(X, y)
        print("Le classifieur a été généré avec succès !", '\n')
        end = time.time()
        print("Temps de calcul de la génération : ", end - start, " secondes")

        ypred, ytrue = functions.test(clf, n_test, df_test)
        M = functions.analyse(ypred, ytrue)
        M.to_csv("Matrice de confusion KNN.csv")

