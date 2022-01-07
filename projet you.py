#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt
import skimage
import skimage.io
import skimage.transform
from PIL import Image
import time

# Extract subset of data
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import functions

start = 1
end = 20

# load dataset
df= pd.read_csv("./csv/chinese_mnist.csv", low_memory = False)
df.head()

print("dataframe rows:", df.shape[0])
print("image files :", len(os.listdir("./Data You/")))

# Matchin image names
def file_path_col(df):
    file_path = f"input_{df[0]}_{df[1]}_{df[2]}.jpg"
    return file_path

# Create file_path column
df["file_path"] = df.apply(file_path_col, axis = 1)
df.head()

# Make training set
df_train = df.loc[(df['suite_id'] >= start) &  (df['suite_id'] < end),:]
X = []
y = []
for i in range(len(df_train)):
    image = skimage.io.imread('./Data You/data/' + df_train['file_path'].values[i])
    image = skimage.transform.resize(image, (64, 64, 1), mode='reflect')
    image = image.reshape(64*64)

    X.append(image)
    y.append(df_train['value'].values[i])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

print('\n')
print("***************** RANDOM FORESTS ***************************",'\n')

print("Génération du classifieur en cours ...")


clf = svm.SVC()
clf.fit(X_train, y_train)
print("Le classifieur a été généré avec succès !",'\n')

n_error = 0

ypred = []
ytrue = []

print("La machine effectue le test ...")
# printProgressBar(0, n_test, prefix='Progress:', suffix='Complete', length=50)

for i in range(0, len(y_test)):

    # pour suivre la progression du test
    if (i % 100 == 0): print(i)

    # print("Chargement(", int(i / n_test * 100), " %)", end='\r')

    test_sample = X_test[i][:]
    answer = y_test[i]

    prediction = clf.predict([test_sample])
    prediction = prediction[0]

    ypred.append(prediction)
    ytrue.append(answer)

    # time.sleep(0.1)
    # printProgressBar(i, n_test, prefix='Progress:', suffix='Complete', length=50)

    if(prediction != answer):
        n_error += 1


print("l'erreur : ", n_error/len(y_test))


# print("L'erreur de la machine est de  n_erreurs/n_essais  = ", n_error / n_test * 100, "%", '\n', '\n')
end = time.time()
print("Temps de calcul du test : ", end - start, " secondes")


print("ypred = ", ypred)
print("ytrue = ", ytrue)
M = functions.analyse(ypred, ytrue)
M.to_csv("Matrice de confusion RF.csv")



print('\n')
