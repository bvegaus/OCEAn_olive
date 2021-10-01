# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 10:44:55 2021

@author: belen
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import arff

def estandarizacion(data):
    # data = datos sin la clase
    columns = data.columns.tolist()
    data_scaled = pd.DataFrame(StandardScaler().fit_transform(data), columns = columns)
    return data_scaled
    
    
    


###############################################################################
#   Leer datos
#Lectura de las 2 hojas normalizadas
datos_f3 = pd.read_excel('./datos_aceite/datos.xlsx', sheet_name=0)
datos_f2 = pd.read_excel('./datos_aceite/datos.xlsx', sheet_name=2)

#División del conjunto de datos en training y test
train_aux = pd.concat([datos_f3[:327],datos_f2[:234]]).reset_index(drop='True')
val = pd.concat([datos_f3[327:],datos_f2[234:]]).reset_index(drop='True')

# Poner la clase al final, tanto en train_aux como en val
y_train_aux = train_aux.Class.map({
      'E': 'nonLOO',
      'L':'LOO',
      'V': 'nonLOO'
      })


train_aux = train_aux.drop(['Name','Class','Baseline', 'RIP Position', 'RIP Height'], axis = 1)
train_aux_scaled = estandarizacion(train_aux)
train_aux_scaled['Class'] = y_train_aux 

y_val = val.Class.map({
    'E': 'nonLOO',
    'L': 'LOO',
    'V': 'nonLOO'
      })

val = val.drop(['Name','Class','Baseline', 'RIP Position', 'RIP Height'], axis = 1)
val_scaled = estandarizacion(val)
val_scaled['Class'] = y_val
arff.dump('./data_weka_LOO_nonLOO/val.arff', val_scaled.values, relation = 'val', names = val_scaled.columns)

for i in range(5):
    train, opt = train_test_split(train_aux_scaled, test_size = val_scaled.shape[0], random_state = i, stratify = train_aux_scaled['Class'])
    arff.dump('./data_weka_LOO_nonLOO/'+str(i)+'.arff', train.values, relation = 'train', names = train.columns)
    arff.dump('./data_weka_LOO_nonLOO/'+str(i+5)+'.arff', opt.values, relation = 'opt', names = opt.columns)
    
    
    


    