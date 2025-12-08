from flask import Flask, request, render_template
import numpy as np
from modelo.medelotfidf import procesar_todo, procesar_query
from modelo.embedding import procesar_todo_emb, procesar_query_emb, normalizar_embeddings

app = Flask(__name__)
print("Procesando TF-IDF y matrices...")

data = procesar_todo()
titles = data["titulos"]
abstract = data["abstract"]
keywords = data["keywords"]
matriz_sim = data["ms_total"]

data_emb = procesar_todo_emb()
titles_emb = data_emb["titulos"]
abstracts_emb = data_emb["abstract"]
keywords_emb = data_emb["keywords"]
matriz_sim_emb = data_emb["ms_total"]
emmbeddings_norm = data_emb["embeddings_norm"]

print("Listo. Sistema cargado.")


# 
# ===========BÚSQUEDA TF-IDF===============
def buscar_documentos(query, top_k=10):
    similitudes = procesar_query(query)

    similitudes = np.asarray(similitudes).flatten()
    
    # ordenar documentos de mayor similitud a menor
    indices = np.argsort(similitudes)[::-1]

    resultados = []
    for i in indices[:top_k]:
        sim_doc_actual = matriz_sim[i]
        docs_relacionados_idx = sim_doc_actual.argsort()[::-1]
        
        relacionados=[]
        for idx in docs_relacionados_idx:
            if idx != i and len(relacionados) < 3:  # los 3 más similares, excluyendo el mismo documento
                relacionados.append({
                    "id": int(idx),
                    "titulo": titles[idx],
                    "score": float(sim_doc_actual[idx])
                })
        resultados.append({
            "id": int(i),
            "titulo": titles[i],
            "abstract": abstract[i],
            "keyword": keywords[i],
            "score": float(similitudes[i]),
            "relacionados": relacionados
        })

    return resultados

# ===========BÚSQUEDA EMBEDDING===============
def buscar_documentos_embedding(query, top_k=10):
    similitudes = procesar_query_emb(query, normalizar_embeddings(emmbeddings_norm))

    # ordenar documentos de mayor similitud a menor
    indices = np.argsort(similitudes)[::-1]
    print(f"DEBUG: Indices: {indices.shape}")

    resultados = []
    for i in indices[:10]:
        sim_doc_actual = matriz_sim_emb[i]
        docs_relacionados_idx = sim_doc_actual.argsort()[::-1]
        
        relacionados=[]
        for idx in docs_relacionados_idx:
            if idx != i and len(relacionados) < 3:  # los 3 más similares, excluyendo el mismo documento
                relacionados.append({
                    "id": int(idx),
                    "titulo": titles_emb[idx],
                    "score": float(sim_doc_actual[idx])
                })
        resultados.append({
            "id": int(i),
            "titulo": titles_emb[i],
            "abstract": abstracts_emb[i],
            "keyword": keywords_emb[i],
            "score": float(similitudes[i]),
            "relacionados": relacionados
        })

    return resultados


# RUTAS FLASK
# ==========================
@app.route("/")
def index():
    return render_template("index.html", documentos=titles)

@app.route("/buscador")
def buscador():
    return render_template("buscador.html")

@app.route("/buscar", methods=["POST"])
def buscar():
    query = request.form["consulta"]
    #print(f"DEBUG: Consulta recibida: {query}")

    resultados = buscar_documentos(query)
    return render_template("resultados.html",
                           consulta=query,
                           resultados=resultados)

@app.route("/buscador_embedding")
def buscador_embedding():
    return render_template("buscadorEmb.html")

@app.route("/buscar_embedding", methods=["POST"])
def buscar_embedding():
    query = request.form["consulta"]
    #print(f"DEBUG: Consulta recibida: {query}")
    resultados = buscar_documentos_embedding(query, emmbeddings_norm)
    return render_template("resultados.html",
                           consulta=query,
                           resultados=resultados)


if __name__ == "__main__":
    app.run(port=5001, debug=True)
