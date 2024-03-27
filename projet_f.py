#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:22:09 2024

@author: cytech
"""
#pip install mysql-connector-python
#pip install codecarbon

import mysql.connector
import csv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import string, re
from bs4 import BeautifulSoup    
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS as stopwords
from codecarbon import EmissionsTracker
import warnings
from bs4 import MarkupResemblesLocatorWarning
import sys

# Charger le modèle spaCy en français
nlp = spacy.load("fr_core_news_sm")
    
# Convertit un fichier CSV en dictionnaire.
# Entrée : chemin_fichier (str) - Chemin vers le fichier CSV sans en-tête.
# Sortie : dictionnaire (dict) - Dictionnaire avec la première colonne comme clés et la seconde comme valeurs.
def lire_csv_vers_dictionnaire(chemin_fichier):
    """
    Lit un fichier CSV et renvoie un dictionnaire à partir des données.

    Args:
        chemin_fichier (str): Chemin vers le fichier CSV.

    Returns:
        dict: Dictionnaire construit à partir des données CSV.
    """
    data = pd.read_csv(chemin_fichier, header=None)
    return dict(zip(data.iloc[:, 0], data.iloc[:, 1]))



# Nettoie une phrase en retirant le HTML, le CSS, les stopwords et la ponctuation.
# Entrée : phrase (str) - La phrase à nettoyer.
# Sortie : clean_words (str) - La phrase nettoyée, en minuscules et sans mots non significatifs.
def clean_phrase(phrase):
    text_without_html = remove_html_tags(phrase)
    text_without_css = remove_css(text_without_html)
    doc = nlp(text_without_css)
    clean_words = ""
    for token in doc:
        if (token.text not in stopwords) and (token.lemma_ not in stopwords) and (token.pos_ != "PUNCT"):
            clean_words += f"{token.text.lower()} "
    return clean_words

# Nettoie et prétraite les données d'un DataFrame selon des dictionnaires de validation.
# Entrée : 
#   dataframe (pd.DataFrame) - Le DataFrame à nettoyer.
#   dictionnaire_solution (dict) - Clés des solutions valides.
#   dictionnaire_techno (dict) - Clés des technologies valides.
# Sortie : 
#   data_clean (pd.DataFrame) - DataFrame nettoyé avec textes prétraités.
def nettoyer_donnees(dataframe, dictionnaire_solution, dictionnaire_techno):
    """
    Nettoie les données dans un DataFrame en supprimant les entrées non valides et en prétraitant les textes.
    
    Args:
        dataframe (pd.DataFrame): DataFrame contenant les données à nettoyer.
        dictionnaire_solution (dict): Dictionnaire contenant les solutions valides.
        dictionnaire_techno (dict): Dictionnaire contenant les technologies valides.
    
    Returns:
        pd.DataFrame: DataFrame nettoyé et prétraité.
    """
    
    indices_a_supprimer = []
    for index, row in dataframe.iterrows():
        if row[0] == "sol" and row[1] not in dictionnaire_solution.keys():
            indices_a_supprimer.append(index)
        elif row[0] == "tec" and row[1] not in dictionnaire_techno.keys():
            indices_a_supprimer.append(index)
    dataframe.drop(indices_a_supprimer, inplace=True)
    dataframe[3] = ''
    nbligne = dataframe.shape[0]
    for i in range(nbligne):
        if dataframe.iloc[i, 0] == "tec":
            if dataframe.iloc[i, 1] in dictionnaire_techno:
                dataframe.iloc[i, 3] = dictionnaire_techno[dataframe.iloc[i, 1]]
        elif dataframe.iloc[i, 0] == "sol":
            if dataframe.iloc[i, 1] in dictionnaire_solution:
                dataframe.iloc[i, 3] = dictionnaire_solution[dataframe.iloc[i, 1]]
         
    
    data_clean = pd.DataFrame()
    data_clean[0] = dataframe[0]
    data_clean[1] = dataframe[1]
    data_clean[2] = dataframe[2].apply(clean_phrase)
    data_clean[3] = dataframe[3]
    return data_clean

# Supprime les balises HTML d'une chaîne de texte.
# Entrée : text (str) - La chaîne de texte contenant des balises HTML.
# Sortie : cleaned_text (str) - La chaîne de texte nettoyée sans balises HTML.
def remove_html_tags(text):
  # Ignorer l'avertissement spécifique de BeautifulSoup
  with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
    # Utiliser BeautifulSoup pour supprimer les balises HTML
    soup = BeautifulSoup(text, "html.parser")
  cleaned_text = soup.get_text()
  return cleaned_text

# Supprime les styles CSS d'une chaîne de texte HTML.
# Entrée :  text (str) - La chaîne de texte HTML contenant les styles CSS.
# Sortie :  cleaned_text (str) - La chaîne de texte HTML nettoyée sans les styles CSS.
def remove_css(text):
    # Utiliser une expression régulière pour supprimer les styles CSS
    cleaned_text = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', text)
    return cleaned_text

# Fusionne les textes associés à une même technologie dans un DataFrame.
# Args: df (pd.DataFrame): Le DataFrame contenant les données.
# Returns:  pd.DataFrame: Le DataFrame avec les textes fusionnés.
def fusionner_textes(df):
    """
    Fusionne les textes associés à une même technologie.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
    
    Returns:
        pd.DataFrame: DataFrame avec les textes fusionnés.
    """
    techMere = []
    for index, row in df.iterrows():
        if (row[0] == "tec") and (row[3] == 1):
            techMere.append((row[2], row[1]))
        elif (row[0] == "tec") and (row[3] != 1):
            codeFils = row[3]
            for txt, code in techMere:
                if code == codeFils:
                    df.loc[index, 2] = txt + " " + row[2]
                    techMere.append((df.loc[index, 2], row[1]))
    df[2] = df[2].fillna('')
    return df

# Obtient les listes de solutions et de technologies uniques à partir d'un DataFrame.
# Args:
#   df (pd.DataFrame): Le DataFrame contenant les données.
# Returns:
#   list: Liste des solutions uniques.
#   list: Liste des technologies uniques.
def obtenir_listes(df):
    """
    Obtient les listes de solutions et de technologies uniques.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
    
    Returns:
        list: Liste des solutions uniques.
        list: Liste des technologies uniques.
    """
    nbSol,nbTec = [],[]
    nbSol = [row[1] for index, row in df.iterrows() if row[0] == "sol" and row[1] not in nbSol]
    nbTec = [row[1] for index, row in df.iterrows() if row[0] == "tec" and row[1] not in nbTec]
    return nbSol, nbTec

# Initialise le vectoriseur TF-IDF à partir d'une liste de textes.
# Args:
#   texts (list): Liste des textes à vectoriser.
# Returns:
#   TfidfVectorizer: Vectoriseur TF-IDF initialisé.
#   sparse matrix: Matrice de vecteurs TF-IDF.
def initialiser_vectorizer(texts):
    """
    Initialise le vectoriseur TF-IDF.
    
    Args:
        texts (list): Liste des textes à vectoriser.
    
    Returns:
        TfidfVectorizer: Vectoriseur TF-IDF initialisé.
        sparse matrix: Matrice de vecteurs TF-IDF.
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    return vectorizer, X

# Récupère une liste de codes de solutions pour une technologie donnée.
# Args:
#   res (list): Liste actuelle des codes de solutions.
#   techno (str): Technologie pour laquelle on recherche les solutions.
#   codes (list): Liste des codes.
#   nature (list): Liste des natures.
#   parent (list): Liste des parents.
# Returns:
#   list: Liste mise à jour des codes de solutions.
def liste_solution(res, techno, codes, nature, parent):
    """
    Récupère une liste de codes de solutions pour une technologie donnée.

    Args:
        res (list): Liste actuelle des codes de solutions.
        techno (str): Technologie pour laquelle on recherche les solutions.
        codes (list): Liste des codes.
        nature (list): Liste des natures.
        parent (list): Liste des parents.

    Returns:
        list: Liste mise à jour des codes de solutions.
    """
    for i in range(len(codes)):
        if nature[i] == "sol" and parent[i] == techno and codes[i] not in res:
            res.append(codes[i])
        elif nature[i] == "tec" and parent[i] == techno:
            res = liste_solution(res, codes[i])
    return res

# Trouve le code correspondant à une question donnée.
# Args:
#   question (str): Question posée par l'utilisateur.
#   vectorizer (TfidfVectorizer): Vectoriseur TF-IDF.
#   X (sparse matrix): Matrice de vecteurs TF-IDF.
#   codes (list): Liste des codes.
#   nature (list): Liste de la nature des codes.
#   parent (list): Liste des parents.
# Returns:
#   str or list: Code ou liste de codes correspondant à la question.
def trouver_code(question, vectorizer, X, codes, nature, parent):
    """
    Trouve le code correspondant à une question donnée.

    Args:
        question (str): Question posée par l'utilisateur.
        vectorizer (TfidfVectorizer): Vectoriseur TF-IDF.
        X (sparse matrix): Matrice de vecteurs TF-IDF.
        codes (list): Liste des codes.
        nature (list): Liste de la nature des codes.

    Returns:
        str or list: Code ou liste de codes correspondant à la question.
    """
    #Vectorisation de la question
    question_vec = vectorizer.transform([question])

    #Calcul des similarités cosinus entre la question et les textes de la base de données
    similarities = cosine_similarity(question_vec,X)

    #Trouver l'indice de la similarité maximale
    max_index = np.argmax(similarities)

    if nature[max_index] == "sol":
      return codes[max_index]
    elif nature[max_index]=="tec":
      return liste_solution([],codes[max_index],codes,nature,parent)

# Extrait les données d'une requête SQL et les enregistre dans un fichier CSV.
# Args:
#   querySQL (str): Requête SQL à exécuter.
#   nameCSV (str): Nom du fichier CSV à créer.
# Returns:
#   int: 0 si l'extraction et l'enregistrement sont réussis, sinon une erreur est affichée.
def ExtractCSV(querySQL, nameCSV):
  try:
    #CONNECTION WITH PERSONAL ID session
    connection = mysql.connector.connect(host='localhost', database='DATABATTLE', user='projet', password='Password1!') 

    if connection.is_connected():
      cursor = connection.cursor()
      
      # Execute the query
      cursor.execute(querySQL)
      records = cursor.fetchall()

      #Create the CSV 
      csv_file_path1 = nameCSV 
      with open(csv_file_path1, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(records)

      #print(f"The data from the first query was successfully exported to {csv_file_path1}")
        
  #ERROR
  except mysql.connector.Error as e:
      print("MySQL connection error", e)

  #CLOSE THE CONNECTION'
  finally:
    if connection.is_connected():
      cursor.close()
      connection.close()
      #print("MySQL connection is closed")
      
  return 0

# Fonction principale pour traiter une requête.
# Args:
#   prompt (str): Question posée par l'utilisateur.
# Returns:
#   None
def main(prompt):
    
    #Extract CSV File
    query1 = "SELECT `typedictionnaire`, `codeappelobjet`, `traductiondictionnaire` FROM `tbldictionnaire` WHERE (`typedictionnaire` = 'tec' OR `typedictionnaire`='sol') AND `codelangue`=2 AND `indexdictionnaire`=1 ORDER BY `tbldictionnaire`.`codeappelobjet` ASC"
    csv_file_path1 = 'tbldictionnaire.csv'
    ExtractCSV(query1, csv_file_path1)
    query2 = "SELECT `numsolution`,`codetechno` FROM `tblsolution` WHERE `codetechno`!=1 AND `codetechno` NOT IN (SELECT tblhidetechno.codetechno FROM tblhidetechno WHERE tblhidetechno.idusergroup=2)"
    csv_file_path2 = 'tblsolution.csv'
    ExtractCSV(query2, csv_file_path2)
    query3 = "SELECT `numtechno`, `codeparenttechno` FROM `tbltechno` WHERE `numtechno`!=1 and `numtechno` not in (SELECT tblhidetechno.codetechno FROM tblhidetechno WHERE tblhidetechno.idusergroup=2)"
    csv_file_path3 = 'tbltechno.csv'
    ExtractCSV(query3, csv_file_path3)

    # Charger le modèle spaCy en français
    nlp = spacy.load("fr_core_news_sm")
    
    chemin_fichier_solution = 'tblsolution.csv'
    chemin_fichier_techno = 'tbltechno.csv'

    dictionnaire_solution = lire_csv_vers_dictionnaire(chemin_fichier_solution)
    dictionnaire_techno = lire_csv_vers_dictionnaire(chemin_fichier_techno)

    chemin_fichier = 'tbldictionnaire.csv'
    data = pd.read_csv(chemin_fichier, header=None)

    data_cleaned = nettoyer_donnees(data, dictionnaire_solution, dictionnaire_techno)

    df_merged = fusionner_textes(data_cleaned)

    list_solutions, list_technologies = obtenir_listes(df_merged)

    codes = df_merged[1].tolist()
    texts = df_merged[2].tolist()
    nature = df_merged[0].tolist()
    parent = df_merged[3].tolist()

    vectorizer, X = initialiser_vectorizer(texts)


    question = prompt
    code_solution = trouver_code(question, vectorizer, X, codes, nature,parent)

    if isinstance(code_solution, list):
        for i in range(len(code_solution)):
            print("{}".format(code_solution[i]), end=",")
    else:
        print("{}".format(code_solution))
        

if __name__ == "__main__":
    # tracker = EmissionsTracker()
    # tracker.start()
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    main(prompt)
    # emissions = tracker.stop()
    # print(emissions)