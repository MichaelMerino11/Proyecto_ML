import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
import time

dataframe = pd.read_csv('modelo/Papers.csv', encoding='latin-1')
dataframe.head()

titles = dataframe['title'].values
#print(titles.shape)
keywords = dataframe['keywords'].values
#print(titles.shape)
abstract = dataframe['abstract'].values
#print(abstract.shape)

#=========================== Preposesamiento (NLP) =======================# 
def normalizacion(texto):
  texto_norm = re.sub('[^A-Za-z0-9á-ú]+', ' ', texto)
  return texto_norm
#--
def tokenizacion(texto):
  return texto.split()
#--
def elim_stopwords(texto):
  #stpwords = stopwords.words('spanish')
  stpwords = stopwords.words('english')
  return [word for word in texto if word not in stpwords]
#--
def stemming(texto):
  #stemmer = nltk.SnowballStemmer('spanish')
  stemmer = PorterStemmer()
  return [stemmer.stem(word) for word in texto]
#--
def NLP(coleccion):
  return [stemming(elim_stopwords(tokenizacion(normalizacion(doc))))
          for doc in coleccion]
#======================= Jaccard ======================================#
def similitud_jaccard(coleccion):
  conjuntos = [set(doc) for doc in coleccion]
  tam = len(conjuntos)
  matrizS = np.zeros((tam,tam))

  for i in range(tam):
    for j in range(i, tam):
      inter = len(conjuntos[i] & conjuntos[j])
      uni = len(conjuntos[i] | conjuntos[j])
      matrizS[i,j] = inter/uni if uni else 0
      matrizS[j,i] = matrizS[i,j]
  return matrizS

#============================= TF-IDF ====================================#
#invertes index
def build_inverted_index(documents):
    inv_i = {}
    for doc_id, doc in enumerate(documents,1):
      for token in doc:
        frec = doc.count(token)
        if token not in inv_i:
          inv_i[token] = []
        inv_i[token].append([doc_id, frec])
    return inv_i
#bolsa de palabras
def bolsa_de_palabras(coleccion, num_documents):
  inverte_i = build_inverted_index(coleccion)
  diccionario = list(inverte_i.keys())  # Lista ordenada de tokens
  # Inicializar matriz de ceros, con tantas filas como tokens y columnas como documentos
  matriz_bolsa_palabras = np.zeros((len(diccionario), num_documents), dtype=int)

  for token, datos in inverte_i.items():
    token_index = diccionario.index(token)

    for doc_id, frec in datos:
      matriz_bolsa_palabras[token_index, doc_id -1] = frec

  return matriz_bolsa_palabras

#--TF-IDF--
def TFIDF(M):
    Wtf = np.where(M > 0, 1 + np.log10(M+1), 0)
    df = np.count_nonzero(M, axis=1).reshape(-1, 1)
    idf = np.log10(M.shape[1] / df)
    return Wtf * idf

def mstriz_vectores_unitarios(matriz):
  return matriz/np.linalg.norm(matriz, axis=0)

#============================= Procesos =============================#

def procesar_titulos():
    return similitud_jaccard(NLP(titles))

def procesar_keywords():
    return similitud_jaccard(NLP(keywords))

def procesar_abstract():
    NLP_abs = NLP(abstract)
    bag = bolsa_de_palabras(NLP_abs, len(NLP_abs))
    tfidf = TFIDF(bag)
    unit = mstriz_vectores_unitarios(tfidf)
    return unit.T @ unit

# ======================= MEDIR TIEMPOS ============================ #

#start = time.time(); ms_titulos = procesar_titulos(); time_t = time.time() - start
#start = time.time(); ms_keywords = procesar_keywords(); time_k = time.time() - start
#start = time.time(); ms_abstract = procesar_abstract(); time_a = time.time() - start
#start = time.time(); ms_total = 0.15*ms_titulos + 0.25*ms_keywords + 0.6*ms_abstract; time_s = time.time() - start

# ======================== RESULTADOS =============================== #

#print(f"Tiempo Titulos: {time_t:.4f} s")
#print(f"Tiempo Keywords: {time_k:.4f} s")
#print(f"Tiempo Abstract TF-IDF: {time_a:.4f} s")
#print(f"Tiempo Suma Ponderada: {time_s:.6f} s")
#tiempo_total = time_t + time_k + time_a + time_s
#print(f"Tiempo Total: {tiempo_total:.6f} s")
#---- imprimir matrices ----#
#print("Matriz de similitud de Jaccard:")
#print(procesar_titulos())
#print(procesar_keywords())
#print(procesar_abstract())
#print(ms_total)


def procesar_todo():
    ms_titulos = procesar_titulos()
    ms_keywords = procesar_keywords()
    ms_abstract = procesar_abstract()

    # matriz final ponderada
    ms_total = 0.15 * ms_titulos + 0.25 * ms_keywords + 0.60 * ms_abstract

    return {
        "titulos": titles,
        "keywords": keywords,
        "abstract": abstract,
        "ms_total": ms_total
    }

def jaccard_query_vs_docs(query_tokens, docs):
      resultados = []
      set_query = set(query_tokens)
      for d in docs:
          inter = len(set_query & set(d))
          uni = len(set_query | set(d))
          sim = inter/uni if uni else 0
          resultados.append(sim)
      return np.array(resultados)

# Preprocesamos una sola vez:
NLP_titles = NLP(titles)
NLP_keywords = NLP(keywords)
NLP_abstract = NLP(abstract)

# bolsa de palabras global
bag = bolsa_de_palabras(NLP_abstract, len(NLP_abstract))
tfidf_docs = TFIDF(bag)
unit_docs = mstriz_vectores_unitarios(tfidf_docs)

# vocabulario global
vocab = list(build_inverted_index(NLP_abstract).keys())


def procesar_query(consulta):
    NLP_query = NLP([consulta])[0]

    # similitud Jaccard
    simi_query_ti = jaccard_query_vs_docs(NLP_query, NLP_titles)
    simi_query_kw = jaccard_query_vs_docs(NLP_query, NLP_keywords)

    # construir vector TF-IDF de la consulta
    vector_query = np.zeros((len(vocab), 1))

    for t in NLP_query:
        if t in vocab:
            vector_query[vocab.index(t)] += 1

    # TF-IDF correcto para la consulta
    tf = np.where(vector_query > 0, 1 + np.log10(vector_query), 0)
    
    # Usar el IDF global calculado de los documentos
    df = np.count_nonzero(tfidf_docs, axis=1).reshape(-1, 1)
    idf_global = np.log10(len(NLP_abstract) / df)
    
    tfidf_q = tf * idf_global
    norm = np.linalg.norm(tfidf_q)
    unit_q = tfidf_q / norm if norm else tfidf_q

    simi_query_abs = (unit_docs.T @ unit_q).flatten()
    # combinación ponderada
    sim_total = 0.15 * simi_query_ti + 0.25 * simi_query_kw + 0.60 * simi_query_abs
    return sim_total

