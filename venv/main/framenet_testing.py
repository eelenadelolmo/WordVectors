import nltk
nltk.download('framenet_v17')
from nltk.corpus import framenet as fn
from googletrans import Translator

translator = Translator()
sentences = ['The quick brown fox', 'jumps over', 'the lazy dog']

for sentence in sentences:
    translation = translator.translate(text=sentence, src='en', dest='es')
    print(translation.origin, ' -> ', translation.text)

print(fn.frames('(?i)creat'))
