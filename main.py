from pathlib import Path
import pandas as pd
import streamlit as st
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_tabular
from fastapi import FastAPI, Response
app = FastAPI()

chemin_models = "./models/"

def load_data():
    chemin = 'C:/Users/nisae/OneDrive/Documents/jupyter_notebook/P7_Poitier_Nicolas/Dashboard/'
    chemin = 'https://raw.githubusercontent.com/Npoitier/Implementez_un_modele_de_scoring/main/'

    Liste_des_prets = chemin + 'data/Dashboard_submitt_values.csv'
    data = pd.read_csv(Liste_des_prets, index_col=0, encoding ='utf-8')
    target = data[['TARGET']].copy()
    list_columns = data.columns.tolist()
    list_columns = [col for col in list_columns if col != 'TARGET']
    data = data[list_columns].copy() 
    return chemin, data, target

def load_preprocessing(chemin, model_name):
    #filename = chemin + 'average_precision_score/' + 'TL_SN_pipe'+model_name+'_final_preprocess_model.sav'
    filename = chemin_models+'TL_SN_pipe'+model_name+'_final_preprocess_model.sav'
    model = pickle.load(open(filename, 'rb'))
    return model

def load_model(chemin, model_name):
    #file = open(chemin + 'data/' + model_name +'_TL_SN_pipe_final_colonnes.csv', "r")
    #file = pd.read_csv(chemin + 'data/' + model_name +'_TL_SN_pipe_final_colonnes.csv', header=0, encoding ='utf-8')
    file = pd.read_csv(chemin + 'data/' + model_name +'_TL_SN_pipe_final_colonnes.csv', header=None, names=['col_name'], encoding ='utf-8')
    features = pd.Series(file['col_name']).tolist()

    #features = []
    #for line in file :    
    #    features.append(line.replace('\n',''))
    if 	model_name == 'RandomForestClassifier':
        seuil = 0.1
    else :
        seuil = 0.5

    #filename = chemin + 'average_precision_score/'+'TL_SN_pipe' + model_name +'_final_model.sav''./' +'/average_precision_score/'+ 
    filename = chemin_models+'TL_SN_pipe' + model_name +'_final_model.sav'
    model = pickle.load(open(filename, 'rb'))

    return model, features, seuil
    
def target_score(target,id_pret):
    classe = target[target.index == int(id_pret)]
    classe = classe.iloc[0].item()
    return classe

def prediction(model_name, id_pret):

    chemin, data, target = load_data()
    
    model, features, seuil = load_model(chemin, model_name)
    X=data[features].copy()    
    preproc = load_preprocessing(chemin, model_name)
    X_transform = preproc.transform(X[X.index == int(id_pret)])
    list_colonnes = preproc.get_feature_names_out().tolist()
    list_colonnes = pd.Series(list_colonnes).str.replace('quanti__','').str.replace('remainder__','').str.replace('quali__','').tolist()
    X_transform = pd.DataFrame(X_transform,columns=list_colonnes)
    X_transform = X_transform.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x)) 
    predic_classe = model.predict_proba(X_transform)[:,1]
    score = (predic_classe > seuil).astype(int)
    
    classe = target_score(target,id_pret)
    
    return int(score),classe

@app.get("/")
def hello():
    return {"message":"Hello you"}
    
@app.get("/predict/{model}{id}")
def predict(model: str, id: int, response: Response):
    score,classe = prediction(model_name, id_pret)
    # si non trouvé
    #response.status_code = 404
    
    predict_value = {'score':score,'classe':classe}
    
    return predict_value
    
#@app.get("/shap/{id}")    
#async def shap(id : int, response: Response):
#    return features_dictionary
    
#@app.get("/lime/{id}")    
#async def lime(id : int, response: Response):
#    return features_dictionary
    
