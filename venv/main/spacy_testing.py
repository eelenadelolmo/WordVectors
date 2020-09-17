from __future__ import unicode_literals
import spacy

# Loading Spacy Spanish large model from path
model_lg_path = "/home/elena/PycharmProjects/WordVectors/venv/main/Spacy_models/es_core_news_lg-2.3.1/es_core_news_lg/es_core_news_lg-2.3.1"
model_md_path = "/home/elena/PycharmProjects/WordVectors/venv/main/Spacy_models/es_core_news_md-2.3.1/es_core_news_lg/es_core_news_md-2.3.1"
model_sm_path = "/home/elena/PycharmProjects/WordVectors/venv/main/Spacy_models/es_core_news_sm-2.3.1/es_core_news_lg/es_core_news_sm-2.3.1"
nlp = spacy.load(model_lg_path)
# os.system("python -m spacy download es_core_news_sm")

doc = nlp("Solo estoy probando qué tal va el anotador")

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)

