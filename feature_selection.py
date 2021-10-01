# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:32:12 2020

@author: belen
"""

import pandas as pd
from sklearn.feature_selection import f_classif, SelectPercentile
import random


random.seed(2)

datos = pd.read_excel('./datos_aceite/data.xlsx')
y = datos.Class
X = datos.drop(columns = ['Name','Class','Baseline','RIP Position','RIP Height',])
feature_names = list(X.columns.values)

selector = SelectPercentile(f_classif, percentile=70)

X_new = selector.fit_transform(X, y)

mask = selector.get_support() #list of booleans
new_features = [] # The list of your K best features

for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)
        
X_new = pd.DataFrame(X_new, columns = new_features)
X_new['Clase'] = y

X_new.to_csv('./datos_aceite/aceite_feature70.csv')
