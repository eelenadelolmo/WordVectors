import os
import re
import ast
import html
import shutil
import spacy
import numpy as np
import pandas as pd
import scipy.spatial
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree
from gensim.models import KeyedVectors
from sentence_transformers import SentenceTransformer
from gensim.models.wrappers import FastText


# Pending: precedence of comparisons


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
nlp = spacy.load(model_lg_path)
# os.system("python -m spacy download es_core_news_sm")

# Parameters for sentence transformed based on BETO
np.set_printoptions(threshold=100)
BETO_model = SentenceTransformer('BETO_model')

# Parameters for sentence transformed based on ROBERTA
# np.set_printoptions(threshold=100)
# BETO_model = SentenceTransformer('BERT_ROBERTA_model')

# Loading Word2vec model for Spanish
w2vec_models = KeyedVectors.load('w2vec_models/complete.kv', mmap='r')

# Loading Word2vec model for English
# w2vec_models = KeyedVectors.load_word2vec_format('w2vec_models_en/GoogleNews-vectors-negative300.bin', binary=True)

# Loading FastText model for Spanish
FastText_models = FastText.load_fasttext_format('FastText_models/cc.es.300.bin')

# Loading FastText model for English
# FastText_models = FastText.load_fasttext_format('FastText_models/cc.en.300.bin')

# Directory containing the output in XML of the feature generation module
input_dir = '/home/elena/PycharmProjects/WordVectors/venv/main/in_xml'

# Directory containing the temporal output in XML of the coreference annotation for whole themes and rhemes
output_dir_tmp = '/home/elena/PycharmProjects/WordVectors/venv/main/out_xml_tmp'
shutil.rmtree(output_dir_tmp, ignore_errors=True)
os.makedirs(output_dir_tmp)

# Directory containing the output in XML of this module (coreference annotator for whole theams and rhematic mentions)
output_dir = '/home/elena/PycharmProjects/WordVectors/venv/main/out_xml_tmp'
shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir)

# Directory containing the output in HTML of this module
output_dir_html = '/home/elena/PycharmProjects/WordVectors/venv/main/out_html'
shutil.rmtree(output_dir_html, ignore_errors=True)
os.makedirs(output_dir_html)

# Directory containing the visual matrix output in png of this module
output_dir_plot = '/home/elena/PycharmProjects/WordVectors/venv/main/out_plot'
shutil.rmtree(output_dir_plot, ignore_errors=True)
os.makedirs(output_dir_plot)



##____________________________________________________________________________________________________________
##____________________________________________________________________________________________________________
##____________________________________________________________________________________________________________
## Getting the themes, rhemes and semantic arguments of the rhemes

for file_xml in os.listdir(input_dir):

    with open(input_dir + '/' + file_xml) as archivo:
        texto = archivo.read().encode(encoding='utf-8').decode('utf-8')
        texto = re.sub('\\& ', '', texto)

    with open(output_dir_tmp + '/' + file_xml, 'w') as nuevo:
        nuevo.write(texto)

    # List composed of the ordered themes sentence in the text
    t_ord = list()

    # List composed of the ordered rhemes sentence in the text
    r_ord = list()

    # List composed of the ordered rhematic main frame arguments for every sentences in the text
    r_ord_sem_roles = list()

    # List composed of the list of  rhematic main frame arguments in the text
    r_ord_sem_roles_all = list()

    # List composed of the tuples with the ordered theme and rheme for every sentence in the text
    t_r_ord = list()

    # List composed of the lists of noun phrases in every rheme
    r_ord_noun_phrases = list()

    # List composed of noun phrases in every rheme
    r_ord_noun_phrases_all = list()


    with open(output_dir_tmp + '/' + file_xml) as f_xml:
        xml = f_xml.read().encode(encoding='utf-8').decode('utf-8')
        root = ET.fromstring(xml)

        for sentence in root.iter('sentence'):

            theme = ""
            rheme = ""
            rheme_sem_role = list()

            for child in sentence:
                if child.tag == 'theme':
                    for token in child:
                        theme += token.text.strip() + ' '
                elif child.tag == 'rheme':
                    for token in child:
                        rheme += token.text.strip() + ' '
                if child.tag == 'semantic_roles':
                    for frame in child:
                        if frame.tag == 'main_frame':
                            for arg in frame:
                                rheme_sem_role.append(arg.attrib['dependent'])

            # Data
            t_r_ord.append((theme, rheme))
            t_ord.append(theme)
            r_ord.append(rheme)
            r_ord_sem_roles.append(rheme_sem_role)
            r_ord_sem_roles_all.extend(rheme_sem_role)
            noun_phrases = [sn.text for sn in nlp(rheme).noun_chunks]
            r_ord_noun_phrases.append(noun_phrases)
            r_ord_noun_phrases_all.extend(noun_phrases)



        # Deleting sentences with no theme and rheme matched
        t_r_ord = [x for x in t_r_ord if x[0] != '' and x[1] != '']

        n_sentence = 0
        for n in range(len(t_ord)):
            if t_ord[n_sentence] == '' and r_ord[n_sentence] == '':
                t_ord.pop(n_sentence)
                r_ord.pop(n_sentence)
                r_ord_sem_roles.pop(n_sentence)
                r_ord_noun_phrases.pop(n_sentence)
                n_sentence -= 1
            n_sentence += 1


        ##____________________________________________________________________________________________________________
        ##____________________________________________________________________________________________________________
        ##____________________________________________________________________________________________________________
        ## Getting theme-theme similarity measures


        # print('\n\n\n\n========================================================')
        # print("** Matching themes to themes **")

        themes_ranking = dict()

        # print("\n\n--------------------------------------------------------")
        # print("\nSentence transformers with BETO as a model")

        theme_embeddings = BETO_model.encode(t_ord)

        # Find the closest closest_n sentences of the corpus for each query sentence based on cosine similarity
        closest_n = len(theme_embeddings)
        for theme, theme_embedding in zip(t_ord, theme_embeddings):
            distances = scipy.spatial.distance.cdist([theme_embedding], theme_embeddings, "cosine")[0]

            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])

            # print("\n\n")
            # print("Theme:", theme)
            # print("\nMost similar themes:")

            themes_ranking[theme.strip()] = dict()

            # Not considering the distance with the theme itself (i.e. selecting the list elements from the second one)
            # Undone because caused problems when the same theme are literally repeated along the document
            for idx, distance in results[:closest_n]:
                # print(t_ord[idx].strip(), "(Score: %.4f)" % (1-distance))
                themes_ranking[theme.strip()][t_ord[idx].strip()] = 1-distance

        # print("\n\n--------------------------------------------------------")
        # print("\nWord2vec embeddings\n\n")

        for idx, theme in enumerate(t_ord):
            # print("Theme:", theme)
            # print("Word Mover's distance from other themes:")

            # Copying the list by values not by reference
            others = t_ord[:]
            del others[idx]
            # print("-", theme + ":", w2vec_models.wmdistance(theme, theme))

            # List of normalized Word Movers Distance for every theme
            WMD_others = list()

            for other in others:
                WMD = w2vec_models.wmdistance(theme, other)
                WMD_others.append(WMD)
                # print("-", other + ":", WMD)

            # Normalizing the Word Movers Distance value to a 0-1 range
            # norm = [1 - (float(i) / sum(WMD_others)) for i in WMD_others]
            norm = [1 - (float(i) / 20) for i in WMD_others]
            for n, other in enumerate(others):
                # print("- (norm.)", other + ":", norm[n])
                themes_ranking[theme.strip()][other.strip()] += norm[n]

            # print('\n\n')


        # print("\n\n--------------------------------------------------------")
        # print("\nFastText embeddings\n\n")

        for idx, theme in enumerate(t_ord):
            # print("Theme:", theme)
            # print("Word Mover's distance from other themes:")

            # Copying the list by values not by reference
            others = t_ord[:]
            del others[idx]
            # print("-", theme + ":", FastText_models.wv.wmdistance(theme, theme))

            # List of normalized Word Movers Distance for every theme
            WMD_others = list()

            for other in others:
                WMD = FastText_models.wv.wmdistance(theme, other)
                WMD_others.append(WMD)
                # print("-", other + ":", WMD)

            # Normalizing the Word Movers Distance value to a 0-1 range
            # norm = [1 - (float(i) / sum(WMD_others)) for i in WMD_others]
            norm = [1 - (float(i) / 3) for i in WMD_others]
            for n, other in enumerate(others):
                # print("- (norm.)", other + ":", norm[n])
                themes_ranking[theme.strip()][other.strip()] += norm[n]

            # print('\n\n')




        ##____________________________________________________________________________________________________________
        ##____________________________________________________________________________________________________________
        ##____________________________________________________________________________________________________________
        ## Getting rheme noun phrases-theme similarity measures

        # print('\n\n\n\n========================================================')
        # print("** Matching rhemes to themes **")
        rhemes_themes_ranking = dict()

        # print("\nSentence transformers with BETO as a model")

        rheme_embeddings = BETO_model.encode(r_ord_noun_phrases_all)

        # Find the closest closest_n sentences of the corpus for each query sentence based on cosine similarity
        closest_n = len(rheme_embeddings)
        for rheme, rheme_embedding in zip(r_ord_noun_phrases_all, rheme_embeddings):
            distances = scipy.spatial.distance.cdist([rheme_embedding], theme_embeddings, "cosine")[0]

            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])

            # print("\n\n")
            # print("Rheme noun phrase:", rheme)
            # print("\nMost similar themes:")

            rhemes_themes_ranking[rheme.strip()] = dict()

            # Not considering the distance with the theme itself (i.e. selecting the list elements from the second one)
            # Undone because caused problems when the same theme are literally repeated along the document
            for idx, distance in results[:closest_n]:
                # print(t_ord[idx].strip(), "(Score: %.4f)" % (1-distance))
                rhemes_themes_ranking[rheme.strip()][t_ord[idx].strip()] = 1-distance


        # print("\n\n--------------------------------------------------------")
        # print("\nWord2vec embeddings\n\n")

        for idx, rheme in enumerate(r_ord_noun_phrases_all):
            # print("Rheme noun phrase:", rheme)
            # print("Word Mover's distance from themes:")

            # Copying the list by values not by reference
            others = t_ord[:]
            # del others[idx]
            # print("-", rheme + ":", w2vec_models.wmdistance(rheme, rheme))

            # List of normalized Word Movers Distance for every theme
            WMD_others = list()

            for other in others:
                WMD = w2vec_models.wmdistance(rheme, other)
                WMD_others.append(WMD)
                # print("-", other + ":", WMD)

            # Normalizing the Word Movers Distance value to a 0-1 range
            # norm = [1 - (float(i) / sum(WMD_others)) for i in WMD_others]
            norm = [1 - (float(i) / 20) for i in WMD_others]
            for n, other in enumerate(others):
                # print("- (norm.)", other + ":", norm[n])
                rhemes_themes_ranking[rheme.strip()][other.strip()] += norm[n]

            # print('\n\n')


        # print("\n\n--------------------------------------------------------")
        # print("\nFastText embeddings\n\n")

        for idx, rheme in enumerate(r_ord_noun_phrases_all):
            # print("Rheme noun phrase:", rheme)
            # print("Word Mover's distance from other themes:")

            # Copying the list by values not by reference
            others = t_ord[:]
            # del others[idx]
            # print("-", rheme + ":", FastText_models.wv.wmdistance(rheme, rheme))

            # List of normalized Word Movers Distance for every theme
            WMD_others = list()

            for other in others:
                WMD = FastText_models.wv.wmdistance(rheme, other)
                WMD_others.append(WMD)
                # print("-", other + ":", WMD)

            # Normalizing the Word Movers Distance value to a 0-1 range
            # norm = [1 - (float(i) / sum(WMD_others)) for i in WMD_others]
            norm = [1 - (float(i) / 3) for i in WMD_others]
            for n, other in enumerate(others):
                # print("- (norm.)", other + ":", norm[n])
                rhemes_themes_ranking[rheme.strip()][other.strip()] += norm[n]

            # print('\n\n')




        ##____________________________________________________________________________________________________________
        ##____________________________________________________________________________________________________________
        ##____________________________________________________________________________________________________________
        ## Getting rheme main semantic frame arguments-rheme main semantic frame arguments similarity measures

        # print('\n\n\n\n========================================================')
        # print("** Matching rhematic main frame arguments to rhematic main frame arguments **")

        rhemes_sr_ranking = dict()

        # print("\n\n--------------------------------------------------------")
        # print("\nSentence transformers with BETO as a model")

        rheme_embeddings = BETO_model.encode(r_ord_sem_roles_all)

        # Find the closest closest_n sentences of the corpus for each query sentence based on cosine similarity
        closest_n = len(rheme_embeddings)
        for rheme, rheme_embedding in zip(r_ord_sem_roles_all, rheme_embeddings):
            distances = scipy.spatial.distance.cdist([rheme_embedding], rheme_embeddings, "cosine")[0]

            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])

            # print("\n\n")
            # print("Rhematic main frame arguments:", rheme)
            # print("\nMost similar rhematic main frame arguments:")

            rhemes_sr_ranking[rheme.strip()] = dict()

            # Not considering the distance with the theme itself (i.e. selecting the list elements from the second one)
            # Undone because caused problems when the same theme are literally repeated along the document
            for idx, distance in results[:closest_n]:
                # print(r_ord_sem_roles_all[idx].strip(), "(Score: %.4f)" % (1-distance))
                rhemes_sr_ranking[rheme.strip()][r_ord_sem_roles_all[idx].strip()] = 1-distance

        # print("\n\n--------------------------------------------------------")
        # print("\nWord2vec embeddings\n\n")

        for idx, rheme in enumerate(r_ord_sem_roles_all):
            # print("Rhematic main frame arguments:", rheme)
            # print("Word Mover's distance from other rhematic main frame arguments:")

            # Copying the list by values not by reference
            others = r_ord_sem_roles_all[:]
            del others[idx]
            # print("-", rheme + ":", w2vec_models.wmdistance(rheme, rheme))

            # List of normalized Word Movers Distance for every theme
            WMD_others = list()

            for other in others:
                WMD = w2vec_models.wmdistance(rheme, other)
                WMD_others.append(WMD)
                # print("-", other + ":", WMD)

            # Normalizing the Word Movers Distance value to a 0-1 range
            # norm = [1 - (float(i) / sum(WMD_others)) for i in WMD_others]
            norm = [1 - (float(i) / 20) for i in WMD_others]
            for n, other in enumerate(others):
                # print("- (norm.)", other + ":", norm[n])
                rhemes_sr_ranking[rheme.strip()][other.strip()] += norm[n]

            # print('\n\n')


        # print("\n\n--------------------------------------------------------")
        # print("\nFastText embeddings\n\n")

        for idx, rheme in enumerate(r_ord_sem_roles_all):
            # print("Rhematic main frame arguments:", rheme)
            # print("Word Mover's distance from other rhematic main frame arguments:")

            # Copying the list by values not by reference
            others = r_ord_sem_roles_all[:]
            del others[idx]
            # print("-", rheme + ":", FastText_models.wv.wmdistance(rheme, rheme))

            # List of normalized Word Movers Distance for every theme
            WMD_others = list()

            for other in others:
                WMD = FastText_models.wv.wmdistance(rheme, other)
                WMD_others.append(WMD)
                # print("-", other + ":", WMD)

            # Normalizing the Word Movers Distance value to a 0-1 range
            # norm = [1 - (float(i) / sum(WMD_others)) for i in WMD_others]
            norm = [1 - (float(i) / 3) for i in WMD_others]
            for n, other in enumerate(others):
                # print("- (norm.)", other + ":", norm[n])
                rhemes_sr_ranking[rheme.strip()][other.strip()] += norm[n]

            # print('\n\n')




        ##____________________________________________________________________________________________________________
        ##____________________________________________________________________________________________________________
        ##____________________________________________________________________________________________________________
        ## Getting rheme noun phrases-rheme noun phrases similarity measures

        # print('\n\n\n\n========================================================')
        # ("** Matching rheme noun phrases to rheme noun phrases **")

        rhemes_ranking = dict()

        # print("\n\n--------------------------------------------------------")
        # print("\nSentence transformers with BETO as a model")

        rheme_embeddings = BETO_model.encode(r_ord_noun_phrases_all)

        # Find the closest closest_n sentences of the corpus for each query sentence based on cosine similarity
        closest_n = len(rheme_embeddings)
        for rheme, rheme_embedding in zip(r_ord_noun_phrases_all, rheme_embeddings):
            distances = scipy.spatial.distance.cdist([rheme_embedding], rheme_embeddings, "cosine")[0]

            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])

            # print("\n\n")
            # print("Rheme noun phrase:", rheme)
            # print("\nMost similar rheme noun phrases:")

            rhemes_ranking[rheme.strip()] = dict()

            # Not considering the distance with the theme itself (i.e. selecting the list elements from the second one)
            # Undone because caused problems when the same theme are literally repeated along the document
            for idx, distance in results[:closest_n]:
                # print(r_ord_noun_phrases_all[idx].strip(), "(Score: %.4f)" % (1-distance))
                rhemes_ranking[rheme.strip()][r_ord_noun_phrases_all[idx].strip()] = 1-distance

        # print("\n\n--------------------------------------------------------")
        # print("\nWord2vec embeddings\n\n")

        for idx, rheme in enumerate(r_ord_noun_phrases_all):
            # print("Rheme noun phrase:", rheme)
            # print("Word Mover's distance from other rheme noun phrases:")

            # Copying the list by values not by reference
            others = r_ord_noun_phrases_all[:]
            del others[idx]
            # print("-", rheme + ":", w2vec_models.wmdistance(rheme, rheme))

            # List of normalized Word Movers Distance for every theme
            WMD_others = list()

            for other in others:
                WMD = w2vec_models.wmdistance(rheme, other)
                WMD_others.append(WMD)
                # print("-", other + ":", WMD)

            # Normalizing the Word Movers Distance value to a 0-1 range
            # norm = [1 - (float(i) / sum(WMD_others)) for i in WMD_others]
            norm = [1 - (float(i) / 20) for i in WMD_others]
            for n, other in enumerate(others):
                # print("- (norm.)", other + ":", norm[n])
                rhemes_ranking[rheme.strip()][other.strip()] += norm[n]

            # print('\n\n')


        # print("\n\n--------------------------------------------------------")
        # print("\nFastText embeddings\n\n")

        for idx, rheme in enumerate(r_ord_noun_phrases_all):
            # print("Rheme noun phrase:", rheme)
            # print("Word Mover's distance from other rheme noun phrases:")

            # Copying the list by values not by reference
            others = r_ord_noun_phrases_all[:]
            del others[idx]
            # print("-", rheme + ":", FastText_models.wv.wmdistance(rheme, rheme))

            # List of normalized Word Movers Distance for every theme
            WMD_others = list()

            for other in others:
                WMD = FastText_models.wv.wmdistance(rheme, other)
                WMD_others.append(WMD)
                # print("-", other + ":", WMD)

            # Normalizing the Word Movers Distance value to a 0-1 range
            # norm = [1 - (float(i) / sum(WMD_others)) for i in WMD_others]
            norm = [1 - (float(i) / 3) for i in WMD_others]
            for n, other in enumerate(others):
                # print("- (norm.)", other + ":", norm[n])
                rhemes_ranking[rheme.strip()][other.strip()] += norm[n]

            # print('\n\n')




        ##____________________________________________________________________________________________________________
        ##____________________________________________________________________________________________________________
        ##____________________________________________________________________________________________________________
        ## Deciding correference sets

        # print("Ranking de correferencias de los temas:")
        # pretty(themes_ranking, indent=1)

        # print("Ranking de correferencias de los sintagmas nominales de los remas con los temas:")
        # pretty(rhemes_themes_ranking, indent=1)

        # print("Ranking de correferencias de los argumentos de los marcos semánticos principales de los remas con los argumentos de los marcos semánticos principales de los remas:")
        # pretty(rhemes_sr_ranking, indent=1)

        # print("Ranking de correferencias de los sintagmas nominales de los remas con los sintagmas nominales de los remas:")
        # pretty(rhemes_ranking, indent=1)

        # Critical value for the weighted semantic similarity to consider two themes corefer to the same underlying concept
        threshold_themes = 2.6

        # List of identifiers for coreference sets
        # ids_coref = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".split()
        ids_coref = ["c_" + str(num) for num in range(200)]

        # Number of coreference sets
        n_sets = -1

        # List of (theme, id) tuples
        theme_id = list()

        # List of themes already included in a corefence set
        coreferent_concepts = list()

        for count, t in enumerate(themes_ranking):

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
                    if themes_ranking[t][t_c] > threshold_themes:
                        theme_id.append((t_c, id_c))
                        coreferent_concepts.append(t_c)

        # print("Thematic coreference analyzed", theme_id)




        # List of (rheme, id) tuples
        rheme_theme_id = list()

        # List of rhemes already included in a corefence set
        agrupado = list()

        # Critical value for the weighted semantic similarity to consider noun phrases in rheme corefer with themes
        threshold_rheme_theme_np = 1.6


        for count, r in enumerate(rhemes_themes_ranking):

            # Assigning the coreference chain id of the current theme to the themes with a semantic similarity measure above the threshold
            for t in rhemes_themes_ranking[r]:
                if r in agrupado:
                    break
                if rhemes_themes_ranking[r][t] > threshold_rheme_theme_np:
                    for e in theme_id:
                        if e[0] == t:
                            id_c = e[1]
                            break
                    agrupado.append(r)
                    rheme_theme_id.append((r, id_c))

            # Not adding new rhematic concepts
            """
            if r not in agrupado:
                n_sets += 1
                id_c = ids_coref[n_sets]
                rheme_theme_id.append((r, id_c))
            """

        # print("Rheme-theme coreference analyzed", rheme_theme_id)




        # List of (rheme, id) tuples
        rheme_sr_id = list()

        # List of rhemes already included in a corefence set
        coreferent_concepts_rheme = list()

        # Critical value for the weighted semantic similarity to consider two rhematic main frame arguments corefer to the same underlying concept
        threshold_rhemes_sr = 1.6


        for count, r in enumerate(rhemes_sr_ranking):

            # If the rheme isn't yet in any coreference chain
            if r not in coreferent_concepts_rheme:
                # Assigning a new id with the next corresponding capital letter
                n_sets += 1
                id_c = ids_coref[n_sets]
                rheme_sr_id.append((r, id_c))
                coreferent_concepts_rheme.append(r)

            # If the theme already is in a coreference chain
            else:
                # Getting the previously assigned id
                for e in rheme_sr_id:
                    if e[0] == r:
                        id_c = e[1]
                        break

            # Assigning the coreference chain id of the current theme to the themes with a semantic similarity measure above the threshold
            for r_c in rhemes_sr_ranking[r]:
                if r_c not in coreferent_concepts_rheme:
                    if rhemes_sr_ranking[r][r_c] > threshold_rhemes_sr:
                        rheme_sr_id.append((r_c, id_c))
                        coreferent_concepts_rheme.append(r_c)

        # print("Rhematic coreference analyzed (main semantic frame arguments)", rheme_sr_id)




        # List of (rheme, id) tuples
        rheme_id = list()

        # List of rhemes already included in a corefence set
        coreferent_concepts_rheme = list()

        # Critical value for the weighted semantic similarity to consider two  two rhematic noun phrases corefer to the same underlying concept
        threshold_rhemes_np = 1.6


        for count, r in enumerate(rhemes_ranking):

            # If the rheme isn't yet in any coreference chain
            if r not in coreferent_concepts_rheme:
                # Assigning a new id with the next corresponding capital letter
                n_sets += 1
                id_c = ids_coref[n_sets]
                rheme_id.append((r, id_c))
                coreferent_concepts_rheme.append(r)

            # If the theme already is in a coreference chain
            else:
                # Getting the previously assigned id
                for e in rheme_id:
                    if e[0] == r:
                        id_c = e[1]
                        break

            # Assigning the coreference chain id of the current theme to the themes with a semantic similarity measure above the threshold
            for r_c in rhemes_ranking[r]:
                if r_c not in coreferent_concepts_rheme:
                    # Matching only rhemes not previously matched with themes
                    if rhemes_ranking[r][r_c] > threshold_rhemes_np and r not in [r_t_coref for r_t_coref, id_coref in rheme_theme_id]:
                        rheme_id.append((r_c, id_c))
                        coreferent_concepts_rheme.append(r_c)


        # Keeping only coreference sets with at least two mentions
        rheme_id_repeated = list()

        # First id of the rhematic coreference sets, updated in order to maintain subsequent id numbers
        id_old = int(rheme_id[0][1].split('_')[1])

        # Dictionary to update other reapeated ids
        to_add = dict()

        for n, x in enumerate(rheme_id):

            # If the rheme has already been included in the list of repeated rhemes
            if x[1] in [ya for ya in to_add]:
                rheme_id_repeated.append((x[0], to_add[x[1]]))
                continue

            # Creating the list of ids of the other rhematic coreference sets
            rheme_id_copy = rheme_id[:]
            rheme_id_copy.pop(n)
            coref_sets_others = [elem[1] for elem in rheme_id_copy]

            # If the rhematic coreference set is repeated
            if x[1] in coref_sets_others:
                rheme_id_repeated.append((x[0], 'c_' + str(id_old)))
                to_add[x[1]] = 'c_' + str(id_old)
                id_old += 1


        # print("Rhematic coreference analyzed (noun phrases)", rheme_id_repeated)




        ##____________________________________________________________________________________________________________
        ##____________________________________________________________________________________________________________
        ##____________________________________________________________________________________________________________
        ## Including conceptual coreference information in the XML

        tree = ElementTree()
        tree.parse(output_dir_tmp + '/' + file_xml)

        for sentence in tree.iter('sentence'):

            # Getting the theme string and the corresponding tag in the XML file
            theme_xml = ""
            rheme_xml = ""

            encontrados_start_end = []

            for child in sentence:
                if child.tag == 'theme':
                    for token in child:
                        theme_xml += token.text.strip() + ' '

                    for e in theme_id:
                        if e[0] == theme_xml.strip():
                            child.set('concept_ref', e[1])

                # Getting all rhematic coreference sets
                rheme_id_all = rheme_theme_id + rheme_sr_id + rheme_id_repeated
                n_corref_int = 1
                id_added_rheme = list()

                if child.tag == 'rheme':

                    for token in child:
                        rheme_xml += token.text.strip() + ' '

                    for e in rheme_id_all:
                        if e[0] in rheme_xml.strip() and e[1] not in id_added_rheme:
                            child.set('concept_ref' + str(n_corref_int), e[1])
                            id_added_rheme.append(e[1])
                            n_corref_int += 1

                    id_added_rheme = list()

                    for e in rheme_id_all:
                        if e[0] in rheme_xml.strip() and e[1] not in id_added_rheme:
                            regex = '</token><token pos="[^>]+?">'.join(e[0].strip().split())
                            regex = '<token pos="[^>]+?">' + regex + '</token>'
                            encontrados_start_end.append([(m.start(0), m.end(0)) for m in re.finditer(regex, ET.tostring(child, encoding="unicode"), re.DOTALL)])
                            encontrados_start_end[-1].append(e[0])
                            encontrados_start_end[-1].append(e[1])
                            # print(regex)
                            # print(ET.tostring(child, encoding="unicode"))
                            # print(encontrados_start_end[-1])
                            id_added_rheme.append(e[1])

                    new_tag = ET.Element('to_annotate')
                    child.append(new_tag)
                    new_tag.text = str(encontrados_start_end)

        tree.write(output_dir_tmp + '/' + file_xml)


        # List of the concept lines to add to the output XML file
        concepts_xml = list()
        id_added = list()

        # Creating every concept tag
        for x in theme_id:
            if x[1] not in id_added:
                id_added.append(x[1])
                concept = '<concept id="' + x[1] + '">' + x[0] + '</concept>'
                concepts_xml.append(concept)

        # Creating every concept tag
        for x in rheme_theme_id:
            if x[1] not in id_added:
                id_added.append(x[1])
                concept = '<concept id="' + x[1] + '">' + x[0] + '</concept>'
                concepts_xml.append(concept)

        # Creating every concept tag
        for x in rheme_sr_id:
            if x[1] not in id_added:
                id_added.append(x[1])
                concept = '<concept id="' + x[1] + '">' + x[0] + '</concept>'
                concepts_xml.append(concept)

        # Creating every concept tag
        for x in rheme_id_repeated:
            if x[1] not in id_added:
                id_added.append(x[1])
                concept = '<concept id="' + x[1] + '">' + x[0] + '</concept>'
                concepts_xml.append(concept)


        # Getting the text of the original XML of the output of the previous module without the DTD
        with open(output_dir_tmp + '/' + file_xml) as in_xml:
            xml_old = in_xml.read()

            from_tag = '(<text id=".+?".+)'
            m = re.search(r'(<text id=".+?\n)((.*\n?)+)', xml_old)

        with open(output_dir_tmp + '/' + file_xml, 'w') as out:

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
            <!ATTLIST rheme concept_ref1 IDREF #IMPLIED>
            <!ATTLIST rheme concept_ref2 IDREF #IMPLIED>
            <!ATTLIST rheme concept_ref3 IDREF #IMPLIED>
            <!ATTLIST rheme concept_ref4 IDREF #IMPLIED>
            <!ATTLIST rheme concept_ref5 IDREF #IMPLIED>
            <!ATTLIST rheme concept_ref6 IDREF #IMPLIED>
            <!ATTLIST rheme concept_ref7 IDREF #IMPLIED>
            <!ATTLIST rheme concept_ref8 IDREF #IMPLIED>
            <!ATTLIST rheme concept_ref9 IDREF #IMPLIED>
            <!ATTLIST rheme concept_ref10 IDREF #IMPLIED>
        <!ELEMENT token (#PCDATA)>
            <!ATTLIST token pos CDATA #REQUIRED>
		<!ELEMENT semantic_roles (frame|main_frame)*>
		<!ELEMENT frame (argument*)>
            <!ATTLIST frame type CDATA #REQUIRED>
            <!ATTLIST frame head CDATA #REQUIRED>
		<!ELEMENT main_frame (argument*)>
            <!ATTLIST main_frame type CDATA #REQUIRED>
            <!ATTLIST main_frame head CDATA #REQUIRED>
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




## Testing unsupervised clustering algorithms

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



## Generating XML with mention annotations

for f in os.listdir(output_dir_tmp):
    with open(output_dir_tmp + '/' + f, 'r') as fich_xml:
        xml_ant = fich_xml.read()
        xml_nue = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>

<!DOCTYPE text [
    <!ELEMENT text (concepts, sentence+)>
        <!ATTLIST text id CDATA #REQUIRED>
    <!ELEMENT concepts (concept+)>
        <!ELEMENT concept (#PCDATA)>
            <!ATTLIST concept id ID #REQUIRED>
    <!ELEMENT sentence (theme, rheme, semantic_roles)>
        <!ELEMENT theme (token*)>
            <!ATTLIST theme concept_ref IDREF #IMPLIED>
        <!ELEMENT rheme (token|mention)*>
            <!ATTLIST rheme concept_ref1 IDREF #IMPLIED>
            <!ATTLIST rheme concept_ref2 IDREF #IMPLIED>
            <!ATTLIST rheme concept_ref3 IDREF #IMPLIED>
            <!ATTLIST rheme concept_ref4 IDREF #IMPLIED>
            <!ATTLIST rheme concept_ref5 IDREF #IMPLIED>
            <!ATTLIST rheme concept_ref6 IDREF #IMPLIED>
            <!ATTLIST rheme concept_ref7 IDREF #IMPLIED>
            <!ATTLIST rheme concept_ref8 IDREF #IMPLIED>
            <!ATTLIST rheme concept_ref9 IDREF #IMPLIED>
            <!ATTLIST rheme concept_ref10 IDREF #IMPLIED>
        <!ELEMENT token (#PCDATA)>
            <!ATTLIST token pos CDATA #REQUIRED>
        <!ELEMENT mention (token+)>
            <!ATTLIST mention concept_ref CDATA #REQUIRED>
		<!ELEMENT semantic_roles (frame|main_frame)*>
		<!ELEMENT frame (argument*)>
            <!ATTLIST frame type CDATA #REQUIRED>
            <!ATTLIST frame head CDATA #REQUIRED>
		<!ELEMENT main_frame (argument*)>
            <!ATTLIST main_frame type CDATA #REQUIRED>
            <!ATTLIST main_frame head CDATA #REQUIRED>
        <!ELEMENT argument EMPTY>
            <!ATTLIST argument type CDATA #REQUIRED>
            <!ATTLIST argument dependent CDATA #REQUIRED>
]>


'''
        xml_nue += re.search(r'(<text.+?)<sentence>', xml_ant, re.DOTALL).group(1)
        sentences_str = re.findall('<sentence>(.+?)</sentence>', xml_ant, re.DOTALL)

        for sentence_str in sentences_str:
            xml_nue += '<sentence>\n\t\t'
            match = re.search('<theme.+?>(.+?)</theme>', sentence_str, re.DOTALL)
            if (match):
                theme_str = match.group(0)
            else:
                theme_str = '<theme>\n\t\t</theme>'

            xml_nue += theme_str + '\n\t\t'

            rheme_str = re.search('</theme>(.+?)<to_annotate>.+?</to_annotate>', sentence_str, re.DOTALL).group(1).strip()
            # Pending to solve
            # to_ann_list = [x for x in ast.literal_eval(re.search('<to_annotate>(.+?)</to_annotate>', xml_ant).group(1))]
            to_ann_list = [x for x in ast.literal_eval(re.search('<to_annotate>(.+?)</to_annotate>', sentence_str).group(1)) if len(x) == 3]

            extra_len = 0
            to_ann_list.sort()
            lista_rangos = [el[0] for el in to_ann_list]

            for tup in to_ann_list:

                # Avoding multiclass mentions
                if lista_rangos.count(tup[0]) == 1:

                    prin = tup[0][0] + extra_len
                    fin = tup[0][1] + extra_len

                    # Avoiding overlapping
                    if '<m' not in rheme_str[prin:fin] and '</m' not in rheme_str[prin:fin] and rheme_str[fin-1] != '<':
                        start_tag = '<mention concept_ref="' + tup[2] + '">'
                        end_tag = '</mention>'
                        extra_len = extra_len + len(start_tag + end_tag)
                        rheme_str = rheme_str[:prin] + start_tag + rheme_str[prin:fin] + end_tag + rheme_str[fin:]

            xml_nue += rheme_str
            xml_nue += '\n\t\t</rheme>\n\t'

            match = re.search('<semantic_roles>(.+?)</semantic_roles>', sentence_str, re.DOTALL)
            if (match):
                semantic_roles_str = match.group(0)
            else:
                semantic_roles_str = '\n\t\t<semantic_roles>\n\t\t</semantic_roles>'

            xml_nue += '\t' + semantic_roles_str + '\n\t</sentence>\n\t'
        xml_nue += '\n</text>'

        with open(output_dir + '/' + f, 'w') as output_dir_nue:
            output_dir_nue.write(xml_nue)




## Generating HTML output and the data for the plot

colors = ['708090', 'D2691E', '556B2F', 'FFE4C4', '000080', '2F4F4F', '4B0082', '8B4513', '6A5ACD', '663399', 'BDB76B', 'FFD700', '00FF7F', 'D2B48C', 'DEB887', '32CD32', 'FFFACD', '0000CD', '008000', 'FFFAF0', '6B8E23', '90EE90', '7B68EE', 'FFFFF0', '5F9EA0', 'FFFFFF', '6495ED', '00CED1', '808080', '00008B', 'FFF8DC', '4169E1', 'FF1493', 'FF6347', 'F4A460', '7FFF00', '808000', 'F5F5DC', '8B008B', 'FFF0F5', 'F0FFF0', '9ACD32', 'ADFF2F', 'FF7F50', 'DC143C', 'FF69B4', 'FFFFE0', 'ADD8E6', 'FFEFD5', '8A2BE2', 'DAA520', '7FFFD4', 'E0FFFF', 'BA55D3', 'FF8C00', '20B2AA', 'AFEEEE', 'B22222', '008080', '2E8B57', 'CD853F', 'B0C4DE', 'FAEBD7', '000000', '228B22', '008B8B', '006400', '8FBC8F', '778899', 'FFDAB9', 'FFFAFA', '696969', 'FFE4B5', 'E9967A', 'F0FFFF', '66CDAA', '800080', '87CEEB', 'D3D3D3', 'C0C0C0', 'FA8072', '4682B4', 'F5F5F5', 'DCDCDC', '00FFFF', '48D1CC', 'B0E0E6', 'FFA07A', 'FF0000', '00FA9A', 'A9A9A9', 'FF4500', 'DDA0DD', 'E6E6FA', 'FFEBCD', 'BC8F8F', 'EE82EE', 'FFA500', 'A0522D', '8B0000', 'F8F8FF', 'FDF5E6', '98FB98', '9370DB', '191970', 'FFF5EE', 'FF00FF', 'EEE8AA', 'FAFAD2', '800000', 'FFC0CB', '9932CC', 'B8860B', '00BFFF', 'FFDEAD', 'FFB6C1', 'DB7093', '00FF00', '40E0D0', 'F5DEB3', 'FFFF00', 'FAF0E6', '3CB371', 'D8BFD8', '9400D3', 'C71585', 'DA70D6', 'F0E68C', '0000FF', 'FFE4E1', 'F5FFFA', 'CD5C5C', '483D8B', '87CEFA', '8B4513', '7CFC00', 'F0F8FF', 'A52A2A', '1E90FF', 'F08080']

for file_xml in os.listdir(output_dir):

    # Every row in the dataframe to plot
    data = list()

    html = """<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <title>Analysis of the theme/rheme coreferences in text</title>
        <style>
            span {white-space:nowrap}
        </style>
    </head>

    <body>
    
        <h1>Conceptos</h1>
    
    """

    with open(output_dir + '/' + file_xml) as xml:

        texto = xml.read()

        root = ET.fromstring(texto)


        ids_colors = dict()
        n_concepts = 0

        for concepts in root.iter('concepts'):
            for concept in concepts:
                ids_colors[concept.attrib['id']] = colors[n_concepts % 140]
                html += '<p style="color:#' + str(colors[n_concepts % 140]) + ';">' + concept.text.strip() + ' (' + concept.attrib['id'] + ')</p>'
                n_concepts += 1

        html += '''

        <h1>Oraciones</h1>'''
        n_oraciones = 0

        for num_sentence, sentence in enumerate(root.iter('sentence')):
            n_oraciones += 1
            theme = ""
            rheme = ""
            rheme_sem_role = list()

            theme_color = 'black'
            theme_id_str = ''
            rheme_color = 'black'
            rheme_id_str = ''

            for child in sentence:
                if child.tag == 'theme':

                    if 'concept_ref' in child.attrib and child.attrib['concept_ref'] in ids_colors:
                        theme_color = ids_colors[child.attrib['concept_ref']]
                        theme_id_str = child.attrib['concept_ref']

                    for token in child:
                        theme += token.text.strip() + ' '



                elif child.tag == 'rheme':

                    for grandson in child:
                        if grandson.tag == 'token':
                            rheme += grandson.text.strip() + ' '
                        if grandson.tag == 'mention':
                            rheme_id_str += grandson.attrib['concept_ref'] + ','
                            rheme_color = ids_colors[grandson.attrib['concept_ref']]
                            rheme += '<span style="white-space:nowrap;color:#' + str(rheme_color) + ';">' + '['
                            for grand_grandson in grandson:
                                rheme += grand_grandson.text.strip() + ' '
                            rheme = rheme[:-1]
                            rheme += ']' + grandson.attrib['concept_ref'] + ' </span>'
                    if len(rheme_id_str) > 1:
                        rheme_id_str = rheme_id_str[:-1]

                    # Colouring the full rheme in the same colour
                    """
                    if 'concept_ref1' in child.attrib and child.attrib['concept_ref1'] in ids_colors:
                        rheme_color = ids_colors[child.attrib['concept_ref1']]
                        rheme_id_str = child.attrib['concept_ref1']

                    for token in child:
                        rheme += token.text.strip() + ' '
                    """
                if child.tag == 'semantic_roles':
                    for frame in child:
                        if frame.tag == 'main_frame':
                            for arg in frame:
                                rheme_sem_role.append(arg.attrib['dependent'])


            html += '<p> [ ' + theme_id_str + ' / ' + rheme_id_str + ' ] <span style="white-space:nowrap;color:#' + str(theme_color) + ';">' + theme + '</span> ////<br/>' + '<nobr>' + rheme + '</nobr>'

            # Colouring the full rheme in the same colour
            # html += '<p> [ ' + theme_id_str + ' / ' + rheme_id_str + ' ] <span style="white-space:nowrap;color:#' + theme_color + ';">' + theme + '</span> ////<br/> <span style="color:#' + rheme_color + ';">' + rheme + '</span></p>'

            # Getting the list of theme and rheme ids
            theme_id_str_list = re.sub("c_", "", theme_id_str).split(",")
            rheme_id_str_list = re.sub("c_", "", rheme_id_str).split(",")

            sent_n_concepts = list(range(n_concepts))

            for tema in theme_id_str_list:
                if tema != "":
                    sent_n_concepts[int(tema)] = 'T'

            for rema in rheme_id_str_list:
                if rema != "":
                    if sent_n_concepts[int(rema)] == 'T':
                        sent_n_concepts[int(rema)] = 'B'
                    else:
                        sent_n_concepts[int(rema)] = 'R'

            for pos, sent_n_concept in enumerate(sent_n_concepts):
                if sent_n_concept not in ['T', 'R', 'B']:
                    sent_n_concepts[pos] = 'N'

            # Appending the sentence id at the beginning
            sent_n_concepts = [num_sentence] + sent_n_concepts
            data.append(sent_n_concepts)

    # print(data)

    html += '\t</body>\n</html>'

    with open(output_dir_html + '/' + file_xml.split('.')[0] + '.html', 'w') as file_html:
        file_html.write(html)

    frases = list()
    conceptos = list()
    tema_rema = list()

    for dato in data:
        frases += [dato[0]] * n_concepts
        conceptos += list(range(n_concepts))
        tema_rema += dato[1:]

    df = pd.DataFrame(dict(n_sentence=frases, n_concept=conceptos, t_r=tema_rema))
    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    # print(df)

    x = df['n_sentence']
    y = df['n_concept']

    colores = {'N': 'white', 'T': 'red', 'R': 'blue', 'B': 'orange'}

    fig, ax = plt.subplots()
    columnas = ['c_' + str(numero) for numero in range(n_concepts)]

    plt.xticks(list(range(n_oraciones)), list(range(n_oraciones)))
    plt.yticks(list(range(n_concepts)), columnas)
    plt.rc('grid', linestyle=":", linewidth='0.5', color='gray')
    plt.grid(True)

    ax.scatter(x, y, c=df['t_r'].map(colores))

    plt.savefig(output_dir_plot + '/' + file_xml.split('.')[0] + '.png')


