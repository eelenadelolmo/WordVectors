import os
import numpy as np
import scipy
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer

np.set_printoptions(threshold=100)
model = SentenceTransformer('BETO_model')

input_dir = '/home/elena/PycharmProjects/WordVectors/venv/main/in_xml'

for file_xml in os.listdir(input_dir):

    # List composed of the ordered themes sentence in the text
    t_ord = list()

    # List composed of the ordered rhemes sentence in the text
    r_ord = list()

    # List composed of the tuples with the ordered theme and rheme for every sentence in the text
    t_r_ord = list()


    with open(input_dir + '/' + file_xml) as f_xml:
        xml = f_xml.read()
        root = ET.fromstring(xml)

        for sentence in root.iter('sentence'):
            for child in sentence:
                if child.tag == 'theme':
                    theme = child.text.strip()
                elif child.tag == 'rheme':
                    rheme = child.text.strip()

            t_r_ord.append((theme, rheme))
            t_ord.append(theme)
            r_ord.append(rheme)

        # Deleting sentences with no theme and rheme matched
        t_r_ord = [x for x in t_r_ord if x[0] != '' and x[1] != '']

        n_sentence = 0
        for n in range(len(t_ord)):
            if t_ord[n_sentence] == '' and r_ord[n_sentence] == '':
                t_ord.pop(n_sentence)
                r_ord.pop(n_sentence)
                n_sentence -= 1
            n_sentence += 1


        print('\n\n\n\n========================================================')


        print("\n\n--------------------------------------------------------")
        print("** Matching themes to themes **")

        theme_embeddings = model.encode(t_ord)

        # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        closest_n = 5
        for theme, theme_embedding in zip(t_ord, theme_embeddings):
            distances = scipy.spatial.distance.cdist([theme_embedding], theme_embeddings, "cosine")[0]

            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])

            print("\n\n")
            print("Theme:", theme)
            print("\nTop 5 most similar themes in the sentence:")

            for idx, distance in results[0:closest_n]:
                print(t_ord[idx].strip(), "(Score: %.4f)" % (1-distance))


        print("\n\n--------------------------------------------------------")
        print("** Matching rhemes to themes **")


        # sentences = t_ord + r_ord
        # sentence_embeddings = model.encode(sentences)
        theme_embeddings = model.encode(t_ord)
        rheme_embeddings = model.encode(r_ord)

        """
        # The result is a list of sentence embeddings as numpy arrays
        for theme, theme_embedding in zip(t_ord, theme_embeddings):
            print("Sentence:", theme)
            print("Embedding:", theme_embedding)
            print("")
        """

        # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        closest_n = 5
        for rheme, rheme_embedding in zip(r_ord, rheme_embeddings):
            distances = scipy.spatial.distance.cdist([rheme_embedding], theme_embeddings, "cosine")[0]

            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])

            print("\n\n")
            print("Rheme:", rheme)
            print("\nTop 5 most similar themes in the sentence:")

            for idx, distance in results[0:closest_n]:
                print(t_ord[idx].strip(), "(Score: %.4f)" % (1-distance))


"""
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
"""