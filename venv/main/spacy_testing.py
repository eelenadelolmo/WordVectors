from __future__ import unicode_literals
import xml.etree.ElementTree as ET
import spacy
from spacy.lang.es import Spanish
from spacy.pipeline import EntityRuler
# from spacy import displacy
import os

# Loading Spacy Spanish large model from path
model_lg_path = "/home/elena/PycharmProjects/WordVectors/venv/main/Spacy_models/es_core_news_lg-2.3.1/es_core_news_lg/es_core_news_lg-2.3.1"
model_md_path = "/home/elena/PycharmProjects/WordVectors/venv/main/Spacy_models/es_core_news_md-2.3.1/es_core_news_lg/es_core_news_md-2.3.1"
model_sm_path = "/home/elena/PycharmProjects/WordVectors/venv/main/Spacy_models/es_core_news_sm-2.3.1/es_core_news_lg/es_core_news_sm-2.3.1"
nlp = spacy.load(model_lg_path)
# os.system("python -m spacy download es_core_news_sm")

# Directory containing the output in XML of the feature generation module
input_dir = '/home/elena/PycharmProjects/WordVectors/venv/main/in_xml'


for file_xml in os.listdir(input_dir):

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

            doc_theme = nlp(theme)
            doc_rheme = nlp(rheme)


            # Morphological annotation (and much more...)
            """
            print("**** Sentence ****")

            print("---- Theme ----")
            for token in doc_theme:
                print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                        token.shape_, token.is_alpha, token.is_stop, token.head.pos_,
            [child for child in token.children])

            print("---- Rheme ----")
            for token in doc_rheme:
                print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                        token.shape_, token.is_alpha, token.is_stop)
            """


            # Entity annotation
            """
            print("---- Theme ----")
            ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc_theme.ents]
            print(ents)

            print("---- Rheme ----")
            ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc_rheme.ents]
            print(ents)
            """


            # EntityRuler (adding entities from rules)
            """
            nlp = Spanish()
            ruler = EntityRuler(nlp)
            patterns = [{"label": "PROBANDO_uni", "pattern": [{"LOWER": "asegurar"}]},
                        {"label": "PROBANDO_multi", "pattern": [{"LOWER": "se"}, {"LOWER": "equivocó"}]}]
            ruler.add_patterns(patterns)
            nlp.add_pipe(ruler)

            print("---- Theme with extra rules for entities ----")
            ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc_theme.ents]
            print(ents)

            print("---- Rheme with extra rules for entities ----")
            ents = [(e.text, e.start_char, e.end_char, e.label_) for e in doc_rheme.ents]
            print(ents)
            """


            # Similarity annotation
            """
            tokens = nlp("perro gato plátano")
            for token1 in tokens:
                for token2 in tokens:
                    print(token1.text, token2.text, token1.similarity(token2))
            """
