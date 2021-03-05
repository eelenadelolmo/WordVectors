from sentence_transformers import SentenceTransformer
import numpy as np

np.set_printoptions(threshold=100)

model = SentenceTransformer('BETO_model')

sentences = ['El presidente anunció las medidas para la siguiente fase.',
             'Posiblemente el presidente anuncia las medidas para la fase tres.',
             'Mi padre es el presidente de la comunidad de la fase tres.']
sentence_embeddings = model.encode(sentences)

# The result is a list of sentence embeddings as numpy arrays
for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")

# Corpus with example sentences
corpus_embeddings = model.encode(sentences)

queries = ['Es el mejor presidente de la escalera.',
           'El presidente no ha encontrado una solución para la siguiente fase.',
           'El presidente anunció las medidas para la siguiente fase.']
query_embeddings = model.encode(queries)

import scipy

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
closest_n = 5
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for idx, distance in results[0:closest_n]:
        print(sentences[idx].strip(), "(Score: %.4f)" % (1-distance))



from sklearn.cluster import KMeans

corpus = [' El Ayuntamiento de Barcelona.',
          ' Los radares.',
          ' El proyecto.',
          ' las cámaras.',
          ' las cámaras.',
          'cada radar.',
          ' así.',
          ' Los responsables del área de Vía Pública.',
          ' en las rondas de Barcelona.',
          ' En la mayoría de los siniestros mortales.',
          ' el automovilista infractor.',
          ' En las rondas , la velocidad máxima autorizada.',
          ' Los responsables del proyecto.',
          ' Para evitar la picaresca',
          ' El conductor.',
          ' los radares.'
          ]

corpus_embeddings = model.encode(corpus)

num_clusters = 4
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])

for i, cluster in enumerate(clustered_sentences):
    print("Cluster ", i+1)
    print(cluster)
    print("")