from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


X, y = make_classification(n_samples=100, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)
print(X)
print(y)

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X, y)

prediction = clf.predict([[0,4,5,7]])

print("prediction = ",prediction )

label_names = {
    0 :"T-shirt/top",
    1 :"Trouser",
    2 :"Pullover",
    3 :"Dress",
    4 :"Coat",
    5 :"Sandal",
    6 :"Shirt",
    7 :"Sneaker",
    8 :"Bag",
    9 :'Ankle boot',
}

indef = 0

print( label_names[indef])