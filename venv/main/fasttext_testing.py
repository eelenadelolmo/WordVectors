from gensim.models.wrappers import FastText

ft = FastText.load_fasttext_format('FastText_models/cc.es.300.bin')
print(ft.wv.most_similar("desgraciada", topn=100))


