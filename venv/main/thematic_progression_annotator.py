import os
import re
import html
import shutil
import numpy as np
# import spacy
import scipy.spatial
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
from gensim.models.wrappers import FastText


# Pretty-printing a dictionary with a n-tabs indentation
def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))


# Loading Spacy Spanish large model from path
model_lg_path = "/home/elena/PycharmProjects/WordVectors/venv/main/Spacy_models/es_core_news_lg-2.3.1/es_core_news_lg/es_core_news_lg-2.3.1"
model_md_path = "/home/elena/PycharmProjects/WordVectors/venv/main/Spacy_models/es_core_news_md-2.3.1/es_core_news_lg/es_core_news_md-2.3.1"
model_sm_path = "/home/elena/PycharmProjects/WordVectors/venv/main/Spacy_models/es_core_news_sm-2.3.1/es_core_news_lg/es_core_news_sm-2.3.1"
# nlp = spacy.load(model_lg_path)
# os.system("python -m spacy download es_core_news_sm")

# Parameters for sentence transformed based on BETO
np.set_printoptions(threshold=100)
model = SentenceTransformer('BETO_model')

# Loading Word2vec model for Spanish
word_vectors = KeyedVectors.load('w2vec_models/complete.kv', mmap='r')

# Loading FastText model for Spanish
ft = FastText.load_fasttext_format('FastText_models/cc.es.300.bin')

# Directory containing the output in XML of the feature generation module
input_dir = '/home/elena/PycharmProjects/WordVectors/venv/main/in_xml'

# Directory containing the output in XML of the coreference annotation
output_dir = '/home/elena/PycharmProjects/WordVectors/venv/main/out_xml'
shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir)


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

            theme = ""
            rheme = ""

            for child in sentence:
                if child.tag == 'theme':
                    for token in child:
                        theme += token.text.strip() + ' '
                elif child.tag == 'rheme':
                    for token in child:
                        rheme += token.text.strip() + ' '

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

        themes_ranking = dict()

        print("\n\n--------------------------------------------------------")
        print("** Matching themes to themes **")
        print("\nSentence transformers with BETO as a model")

        theme_embeddings = model.encode(t_ord)

        # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        closest_n = len(theme_embeddings)
        for theme, theme_embedding in zip(t_ord, theme_embeddings):
            distances = scipy.spatial.distance.cdist([theme_embedding], theme_embeddings, "cosine")[0]

            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])

            print("\n\n")
            print("Theme:", theme)
            print("\nMost similar themes in the sentence:")

            themes_ranking[theme.strip()] = dict()

            # Not considering the distance with the theme itself (i.e. selecting the list elements from the second one)
            for idx, distance in results[1:closest_n]:
                print(t_ord[idx].strip(), "(Score: %.4f)" % (1-distance))
                themes_ranking[theme.strip()][t_ord[idx].strip()] = 1-distance

        print("\n\n--------------------------------------------------------")
        print("** Matching themes to themes **")
        print("\nWord2vec embeddings\n\n")

        for idx, theme in enumerate(t_ord):
            print("Theme:", theme)
            print("Word Mover's distance from other themes:")

            # Copying the list by values not by reference
            others = t_ord[:]
            del others[idx]
            print("-", theme + ":", word_vectors.wmdistance(theme, theme))

            # List of normalized Word Movers Distance for every theme
            WMD_others = list()

            for other in others:
                WMD = word_vectors.wmdistance(theme, other)
                WMD_others.append(WMD)
                print("-", other + ":", WMD)

            # Normalizing the Word Movers Distance value to a 0-1 range
            # norm = [1 - (float(i) / sum(WMD_others)) for i in WMD_others]
            norm = [1 - (float(i) / 20) for i in WMD_others]
            for n, other in enumerate(others):
                print("- (norm.)", other + ":", norm[n])
                themes_ranking[theme.strip()][other.strip()] += norm[n]

            print('\n\n')


        print("\n\n--------------------------------------------------------")
        print("** Matching themes to themes **")
        print("\nFastText embeddings\n\n")

        for idx, theme in enumerate(t_ord):
            print("Theme:", theme)
            print("Word Mover's distance from other themes:")

            # Copying the list by values not by reference
            others = t_ord[:]
            del others[idx]
            print("-", theme + ":", ft.wv.wmdistance(theme, theme))

            # List of normalized Word Movers Distance for every theme
            WMD_others = list()

            for other in others:
                WMD = ft.wv.wmdistance(theme, other)
                WMD_others.append(WMD)
                print("-", other + ":", WMD)

            # Normalizing the Word Movers Distance value to a 0-1 range
            # norm = [1 - (float(i) / sum(WMD_others)) for i in WMD_others]
            norm = [1 - (float(i) / 3) for i in WMD_others]
            for n, other in enumerate(others):
                print("- (norm.)", other + ":", norm[n])
                themes_ranking[theme.strip()][other.strip()] += norm[n]

            print('\n\n')


        ## Including conceptual coreference information in the XML

        print("Ranking de correferencias:")
        pretty(themes_ranking, indent=1)

        # Critival value for the weighted semantic similarity to consider two themes corefer to the same underlying concept
        threshold = 1.6

        # List of themes already included in a corefence set
        coreferent_concepts = list()

        # List of identifiers for coreference sets
        ids_coref = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".split()

        # Number of coreference sets
        n_sets = -1

        # List of (theme, id) tuples
        theme_id = list()

        for t in themes_ranking:

            # If the theme isn't yet in any coreference chain
            if t not in coreferent_concepts:
                # Assigning a new id with the next corresponding capital letter
                n_sets += 1
                id_c = ids_coref[n_sets]
                theme_id.append((t, id_c))
                coreferent_concepts.append(t)

            # If the theme already is in a coreference chain
            else:
                # Getting the previously assigned id
                for e in theme_id:
                    if e[0] == t:
                        id_c = e[1]
                        break

            # Assigning the coreference chain id of the current theme to the themes with a semantic similarity measure above the threshold
            for t_c in themes_ranking[t]:
                if t_c not in coreferent_concepts:
                    if themes_ranking[t][t_c] > threshold:
                        theme_id.append((t_c, id_c))
                        coreferent_concepts.append(t_c)


        print("Coreference analyzed", theme_id)



        tree = ElementTree()
        tree.parse(input_dir + '/' + file_xml)

        for sentence in tree.iter('sentence'):

            # Getting the theme string and the corresponding tag in the XML file
            theme_xml = ""
            for child in sentence:
                if child.tag == 'theme':
                    for token in child:
                        theme_xml += token.text.strip() + ' '

                    for e in theme_id:
                        if e[0] == theme_xml.strip():
                            child.set('concept_ref', e[1])

        tree.write(input_dir + '/' + file_xml)


        # List of the concepts lines to add to the output XML file
        concepts_xml = list()
        id_added = list()

        # Creating every concept tag
        for x in theme_id:
            if x[1] not in id_added:
                id_added.append(x[1])
                concept = '<concept id="' + x[1] + '">' + x[0] + '</concept>'
                concepts_xml.append(concept)


        # Getting the text of the original XML of the output of the previous module without the DTD
        with open(input_dir + '/' + file_xml) as in_xml:
            xml_old = in_xml.read()

            from_tag = '(<text id=".+?".+)'
            m = re.search(r'(<text id=".+?\n)((.*\n?)+)', xml_old)

        with open(output_dir + '/' + file_xml, 'w') as out:

            xml_new = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>

<!DOCTYPE text [
    <!ELEMENT text (concepts, sentence+)>
        <!ATTLIST text id CDATA #REQUIRED>
    <!ELEMENT concepts (concept+)>
        <!ELEMENT concept (#PCDATA)>
            <!ATTLIST concept id ID #REQUIRED>
    <!ELEMENT sentence (theme, rheme, semantic_roles)>
        <!ELEMENT theme (token*)>
            <!ATTLIST theme concept_ref IDREF #REQUIRED>
        <!ELEMENT rheme (token*)>
        <!ELEMENT token (#PCDATA)>
            <!ATTLIST token pos CDATA #REQUIRED>
        <!ELEMENT semantic_roles (frame*)>
        <!ELEMENT frame (argument*)>
            <!ATTLIST frame type CDATA #REQUIRED>
            <!ATTLIST frame head CDATA #REQUIRED>
        <!ELEMENT argument EMPTY>
            <!ATTLIST argument type CDATA #REQUIRED>
            <!ATTLIST argument dependent CDATA #REQUIRED>
]>


'''
            # Getting the first and the rest of the lines of the original XML document in order to add the concepts block between them
            xml_old_root = m.group(1)
            xml_old_rest = m.group(2)

            xml_new += xml_old_root

            xml_new += '\n\n\t' + '<concepts>\n\t\t'

            xml_new += '\n\t\t'.join(concepts_xml)

            xml_new += '\n\t' + '</concepts>'

            xml_new += xml_old_rest

            out.write(html.unescape(xml_new))









        ## Matching rhemes PENDING

        """
        print("\n\n--------------------------------------------------------")
        print("** Matching rhemes to themes **")

        # sentences = t_ord + r_ord
        # sentence_embeddings = model.encode(sentences)
        theme_embeddings = model.encode(t_ord)
        rheme_embeddings = model.encode(r_ord)

        # The result is a list of sentence embeddings as numpy arrays
        for theme, theme_embedding in zip(t_ord, theme_embeddings):
            print("Sentence:", theme)
            print("Embedding:", theme_embedding)
            print("")

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




















## Tesing unsupervised clustering algorithms

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