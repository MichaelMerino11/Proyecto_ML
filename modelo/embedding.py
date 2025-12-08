import numpy as np
import pandas as pd

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

dataframe = pd.read_csv('modelo/Papers.csv', encoding='latin-1')
dataframe.head()

titles = dataframe['title'].values
#print(titles.shape)
keywords = dataframe['keywords'].values
#print(titles.shape)
abstracts = dataframe['abstract'].values
#print(abstracts.shape)


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def normalizar_embeddings(embeddings):
    """Normaliza embeddings para cálculo eficiente de similitud"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # evitar división por cero
    return embeddings / norms

#tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base")
#model = AutoModel.from_pretrained("intfloat/e5-base")
#model.eval()

"""def embed_e5(text):
    inp = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        out = model(**inp).last_hidden_state
    emb = out.mean(dim=1)  # average pooling
    return emb.squeeze().numpy()"""


model = SentenceTransformer("all-MiniLM-L6-v2")

def procesar_todo_emb():
    #embeddings = []

    #for abs_text in abstracts:
     #   emb = embed_e5("passage: " + abs_text)
      #  embeddings.append(emb)

    #embeddings = np.array(embeddings)
    embeddings = model.encode(abstracts, show_progress_bar=True)
    #print(f"DEBUG: Shape de embeddings: {embeddings.shape}")
    embeddings_norm = normalizar_embeddings(embeddings)
    #print(f"DEBUG: Shape de embeddings_norm: {embeddings_norm.shape}")

    matriz_sim = embeddings_norm @ embeddings_norm.T
    #print(f"DEBUG: Shape de matriz_sim: {matriz_sim.shape}")
    
    data = {
        "titulos": titles,
        "abstract": abstracts,
        "keywords": keywords,
        "ms_total": matriz_sim,
        "embeddings_norm": embeddings_norm
    }
    return data

def procesar_query_emb(query, embeddings_norm):
    #query_emb = embed_e5("query: " + query)
    query_emb = model.encode(query)
    #print(f"DEBUG: Shape query_emb: {query_emb.shape}")

    # Asegurar que query_emb sea 1D
    if query_emb.ndim > 1:
        query_emb = query_emb[0]

    query_norm = normalizar_embeddings(np.array([query_emb]))[0]
    #print(f"DEBUG: Shape query_norm: {query_norm.shape}")
    #print(f"DEBUG: Shape embeddings_norm: {embeddings_norm.shape}")
    
    similitudes = embeddings_norm @ query_norm
    similitudes = np.asarray(similitudes).flatten()
    #print(f"DEBUG: Shape similitudes: {similitudes.shape}, {similitudes.dtype}")
    
    return similitudes