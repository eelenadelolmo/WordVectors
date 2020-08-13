from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

annotated_uploaded_path = 'uploads/predicted-args.conll'
SRL = dict()

with open(annotated_uploaded_path, "r+") as f:
    texto = f.read()
    anotaciones = texto.split('\n\n')
    for anotacion in anotaciones:
        lineas = anotacion.split('\n')

        # Getting the frame identificator of the anotacion, which may be after the first labelled role
        for linea in lineas:
            if len(linea) > 1:
                frame_id = linea.split('\t')[-3]
                frame_type = linea.split('\t')[-2]
                if frame_id != '_':
                    anotacion_frame = frame_type
                    SRL[anotacion_frame] = list()
                    break

        n_linea = 0
        for linea in lineas:
            if len(linea) > 1:
                n_linea += 1
                iob = linea.split('\t')[-1]
                SRL_tag = iob[2:]
                str = linea.split('\t')[1]
                if 'B-' in iob:
                    ann_tmp = str
                    lineas_siguientes = lineas[n_linea:]

                    for linea_siguiente in lineas_siguientes:
                        iob_sig = linea_siguiente.split('\t')[-1]
                        str_sig = linea_siguiente.split('\t')[1]
                        if 'I-' in iob_sig:
                            ann_tmp = ann_tmp + " " + str_sig
                        else:
                            SRL[anotacion_frame].append((SRL_tag, ann_tmp))
                            break

print(SRL)

model = SentenceTransformer('distiluse-base-multilingual-cased')
sentence_original = 'El presidente afirmó que la principal causa de la deuda era el aumento del paro en los últimos meses'
sentence_original_tokens = word_tokenize(sentence_original)
sentences_original_all_subphrases = [sentence_original_tokens[i: j] for i in range(len(sentence_original_tokens))
          for j in range(i + 1, len(sentence_original_tokens) + 1)]

sentences = list()
for s in sentences_original_all_subphrases:
    sentences.append(' '.join(s))

sentence_embeddings = model.encode(sentences)

# The result is a list of sentence embeddings as numpy arrays
"""
for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
"""

queries = ['the president',
           'stated',
           'that the main cause of the debt was the increase in unemployment in recent months',
           'cause',
           'of the debt',
           'increase',
           'in unemployment',
           'in recent months']
query_embeddings = model.encode(queries)


import scipy

# Find the closest sentences of the corpus for each query sentence based on cosine similarity
closest_n = 1
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for idx, distance in results[0:closest_n]:
        print(sentences[idx].strip(), "(Score: %.4f)" % (1-distance))

