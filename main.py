from pathlib import Path
import pandas as pd
import streamlit as st
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_tabular
from fastapi import FastAPI, Response
from lightgbm import LGBMClassifier

app = FastAPI()

chemin_models = "./models/"

def get_chemin():
    chemin = 'https://raw.githubusercontent.com/Npoitier/API_modele_de_scoring/main/'
    return chemin

def load_data():
    chemin = 'C:/Users/nisae/OneDrive/Documents/jupyter_notebook/P7_Poitier_Nicolas/Dashboard/'
    chemin = 'https://raw.githubusercontent.com/Npoitier/Implementez_un_modele_de_scoring/main/'
    chemin = 'https://raw.githubusercontent.com/Npoitier/API_modele_de_scoring/main/'
    #chemin = get_chemin()

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

def load_model(chemin, model_name, metric = 'average_precision_score'):
    #file = open(chemin + 'data/' + model_name +'_TL_SN_pipe_final_colonnes.csv', "r")
    #file = pd.read_csv(chemin + 'data/' + model_name +'_TL_SN_pipe_final_colonnes.csv', header=0, encoding ='utf-8')
    file = pd.read_csv(chemin + 'data/' + model_name +'_TL_SN_pipe_final_colonnes.csv', header=None, names=['col_name'], encoding ='utf-8')
    features = pd.Series(file['col_name']).tolist()

    #features = []
    #for line in file :    
    #    features.append(line.replace('\n',''))
    file = 'data/' + 'Seuils.csv'
    df_seuils = pd.read_csv(chemin+file,index_col=0)    
    seuil = float(df_seuils.loc[((df_seuils['Classifier']==model_name)&(df_seuils['metric']==metric)),'seuil'].head(1).item())
    
    #filename = chemin + 'average_precision_score/'+'TL_SN_pipe' + model_name +'_final_model.sav''./' +'/average_precision_score/'+ 
    filename = chemin_models+'TL_SN_pipe' + model_name +metric+'_final_model.sav'
    model = pickle.load(open(filename, 'rb'))

    return model, features, seuil
    
def target_score(target,id_pret):
    classe = target[target.index == int(id_pret)]
    classe = classe.iloc[0].item()
    return classe

def prediction(model_name, id_pret, metric):

    chemin, data, target = load_data()
    
    model, features, seuil = load_model(chemin, model_name, metric)
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
    
    if seuil-float(predic_classe) >= 0 :
        percent_fiabilite = np.abs(seuil-float(predic_classe))/seuil*100
    else : 
        percent_fiabilite = np.abs(seuil-float(predic_classe))/(1-seuil)*100
    
    return int(score),classe, percent_fiabilite

def shap_importance(model_name,id_pret, metric):
    # penser à construire pour l'autre métrique et à différentier les noms
    chemin = 'https://raw.githubusercontent.com/Npoitier/API_modele_de_scoring/main/'
    chemin = get_chemin()
    
    df_shap_values = pd.read_csv(chemin + 'data/' +model_name+metric+"_shap_values.csv",
                                 index_col=0, encoding ='utf-8')
    #height = list(df_shap_values.iloc[id_pret])
    height = df_shap_values[df_shap_values.index == int(id_pret)]
    height = np.array(height.T)
    height = height[:,0].tolist()
    #somme = np.sum(height)
    maxi = np.max(np.abs(height))
    mini = np.min(np.abs(height))
    bars = df_shap_values.columns.tolist()
    bars = [bars[x] for x in range(len(height)) if np.abs(height[x]) >= maxi*0.05]
    height = [height[x] for x in range(len(height)) if np.abs(height[x]) >= maxi*0.05]
    
    features_dictionary = dict()
    for i in range(len(height)):    
        features_dictionary[bars[i]] =height[i]   
    
    return features_dictionary
    
def lime_importance(model_name, id_pret, metric):

    chemin, data, target = load_data()
    
    model, features, seuil = load_model(chemin, model_name, metric)
    X=data[features].copy()   
    preproc = load_preprocessing(chemin, model_name)    
    X_transform = preproc.transform(X) 
    list_colonnes = preproc.get_feature_names_out().tolist()
    list_colonnes = pd.Series(list_colonnes).str.replace('quanti__','').str.replace('remainder__','').str.replace('quali__','').tolist()
    X_transform = pd.DataFrame(X_transform,columns=list_colonnes)
    X_transform = X_transform.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x)) 
    list_colonnes = X_transform.columns.tolist()
    X_transform.index = X.index
    explainer = lime_tabular.LimeTabularExplainer(X_transform, mode="classification",
                                              class_names=["Solvable", "Non Solvable"],
                                              feature_names=list_colonnes,
                                                 discretize_continuous=False)
    expdt0 = explainer.explain_instance(np.array(X_transform[X_transform.index == int(id_pret)].T).ravel(),
                                        model.predict_proba ,num_features=len(list_colonnes))
    test = np.array(expdt0.local_exp.get(1))
    list_cols = [list_colonnes[int(i)] for i in test[:,0]]
    height = test[:,1].tolist() 
    height.reverse()    
    #somme = np.sum(height)
    maxi = np.max(np.abs(height))
    mini = np.min(np.abs(height))
    bars = list_cols
    bars.reverse()
    bars = [bars[x] for x in range(len(height)) if np.abs(height[x]) >= maxi*0.05]
    height = [height[x] for x in range(len(height)) if np.abs(height[x]) >= maxi*0.05]
    
    features_dictionary = dict()
    for i in range(len(height)):    
        features_dictionary[bars[i]] =height[i]   
    
    return features_dictionary

def model_features_importance(model_name, metric):
    
    
    chemin = get_chemin()
    
    model, feat, seuil = load_model(chemin, model_name, metric)
    list_importance = model.steps[0][1].feature_importances_
    # on classe les indices d'importance des features
    importance = list_importance.argsort()
    
    # on stocke le max
    maxi = list_importance[importance[-1]]
    i = -1
    idx = []
    while (list_importance[importance[i]]> maxi*0.05) :
        idx.append(importance[i])
        i -= 1
    values = list_importance[idx]
    
    if hasattr(model.steps[0][1], 'feature_name_'):
        preproc = load_preprocessing(chemin, model_name)
        list_colonnes = preproc.get_feature_names_out().tolist()
        list_colonnes = pd.Series(list_colonnes).str.replace('quanti__','').str.replace('remainder__','').str.replace('quali__','').tolist()
        features = list_colonnes    
    else:        
        features = model.steps[0][1].feature_names_in_[idx]
        
    features_dictionary = dict()
    for i in range(len(values)):    
        features_dictionary[features[i]] =values[i] 
    
    return features_dictionary

@app.get("/")
def hello():
    return {"message":"Hello you"}
    
@app.get("/featureimportance/{model}/metric/{metric}")
def featureimportance(model: str, metric : str, response: Response):
    features_dictionary = model_features_importance(model, metric)
    return features_dictionary

    
@app.get("/predict/{model}/metric/{metric}/indice/{id}")
def predict(model: str, metric : str, id: int, response: Response):
    #metric = 'average_precision_score'
    score,classe,percent_fiabilite = prediction(model, id, metric)
    # si non trouvé
    #response.status_code = 404
    
    predict_value = {'score':score,'classe':classe,'fiabilite':percent_fiabilite}
    
    return predict_value
    
@app.get("/shap/{model}/metric/{metric}/indice/{id}")
async def shap(model: str, metric : str, id: int, response: Response):
    #metric = 'average_precision_score'
    features_dictionary = shap_importance(model,id, metric)
    return features_dictionary
    
@app.get("/lime/{model}/metric/{metric}/indice/{id}")
async def lime(model: str, metric : str, id: int, response: Response):
    #metric = 'average_precision_score'
    features_dictionary = lime_importance(model, id, metric)
    return features_dictionary
    
