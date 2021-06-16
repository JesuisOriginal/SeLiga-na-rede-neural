import numpy as np 
import pandas as pd

import catboost
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import FunctionTransformer, Normalizer

from sklearn.base import TransformerMixin, BaseEstimator

import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('saitama.csv')

X_test = pd.read_csv('X_test.csv', index_col=[0])
X_train = pd.read_csv('X_train.csv', index_col=[0])
y_train = pd.read_csv('y_train.csv', index_col=[0])
y_test = pd.read_csv('y_test.csv', index_col=[0])

predict_test = pd.read_csv('predict_test_1.csv')

print('shape of x train ', X_train.shape)
print('shape of y train ', y_train.shape)

# Criando modelo

## Features do GBT
nyan_feat = []
for col in X_train:
    print(col)
    nyan_feat.append(col)

## Range de parametros

params1 = {
    
   'l2_leaf_reg': [1,2,3,4], # coeff de regularizacao

   'depth': [2,3,4,5], # prfund da arvore

   'bagging_temperature':[0, 1, 2, 3, 4, 5], # intensidade do baggind durante bootiesTrap 🍑

   'rsm': [0.6000000000000001, 0.7000000000000001, 0.8, 0.9, 1.0],

   'learning_rate':[0.5, 0.6, 0.7, 0.8, 0.9],

   'od_pval':[0.1, 0.091, 0.089, 0.07, 0.06]

}  # compart de recursos durante selecao

           


cb1 = CatBoostClassifier(eval_metric='AUC', cat_features=nyan_feat, early_stopping_rounds=20, learning_rate=0.1, task_type=GPU,  thread_count=-1)

search_results = cb1.grid_search(params1, X_train, y_train, cv=3, plot=True, verbose=False)

# Resultado do grid search:  {'bagging_temperature': 0, 'depth': 5, 'l2_leaf_reg': 1, 'od_pval': 0.07, 'rsm': 1.0, 'learning_rate': 0.8}
 
# print(search_results['params'])
# data = search_results['params']
# import json
# with open('params.json', 'w', encoding='utf-8') as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)
results = {'bagging_temperature': 0, 'depth': 5, 'l2_leaf_reg': 1, 'od_pval': 0.07, 'rsm': 1.0, 'learning_rate': 0.8}

print('Resultado do grid search: ', results)

cb_grid = CatBoostClassifier(eval_metric='AUC', cat_features=nyan_feat, early_stopping_rounds=40, learning_rate=0.8, depth=5, task_type='GPU', thread_count=-1,l2_leaf_reg=1, rsm=1.0, bagging_temperature=0, od_pval=0.07)
cb_grid.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features=nyan_feat, use_best_model=True, verbose=True)

cb_default = CatBoostClassifier(eval_metric='AUC', cat_features=nyan_feat, early_stopping_rounds=20, task_type='GPU')
cb_default.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features=nyan_feat, use_best_model=True, verbose=True)

print('validação para o melhor modelo com parâmetros personalizados: ',cb_grid.best_score_['validation']['AUC'])
print('validação para o melhor modelo com parâmetro sem ter sido bulido: ',cb_default.best_score_['validation']['AUC'])

if cb_grid.best_score_['validation']['AUC'] > cb_default.best_score_['validation']['AUC']:
        print('o modelo grid teve melhor desempenho')
        
X_predict_test = predict_test.drop(columns=['opinion'])
Y_pred_test = predict_test['opinion'].copy()

correct = Y_pred_test.to_frame()['opinion'][0]

print('Probabilidade de leitura do 37 sem grif: ', cb_grid.predict_proba(X_predict_test)[:,1][0])
print('Resposta correta ', correct)

noticia = X_test.iloc[5000,:].to_frame().T
print('Probabilidade de leitura: ', cb_grid.predict_proba(noticia)[:,1][0])