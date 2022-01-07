import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import functions
import time
from sklearn.neighbors import KNeighborsClassifier



#Sur combien d'échantillons voulez-vous ENTRAINER votre machine (max 60 000) :
n_train = 60000

#Sur combien d'échantillons voulez-vous TESTER votre machine :  (max 10 000)"
n_test = 10000



#Chargement des données
df_train = pd.read_csv('./Data/fashion-mnist_train.csv')
df_test = pd.read_csv('./Data/fashion-mnist_test.csv')


#on sépare les input et le output
X = df_train.drop(labels = 'label', axis=1).values[0:n_train,:]
y = df_train['label'].values[0:n_train]


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

        print("Génération du classifieur en cours ...")
        start = time.time()

        clf = svm.SVC(kernel = 'rbf')
        clf.fit(X, y)
        print("Le classifieur a été généré avec succès !",'\n')
        end = time.time()
        print("Temps de calcul de la génération : ", end - start, " secondes")

        ypred, ytrue = functions.test(clf,n_test,df_test)
        print("ypred = ",ypred)
        print("ytrue = ", ytrue)

        M = functions.analyse(ypred, ytrue)
        M.to_csv("Matrice de confusion SVM.csv")

        print('\n')


    if(choix == 3):
        print("***************** RESEAUX DE NEURONES ***************************",'\n')

        print("Génération du classifieur en cours ...")
        start = time.time()
        clf = MLPClassifier()
        clf.fit(X, y)
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
        M.to_csv("Matrice de confusion KN.csv")
