from gensim.models import KeyedVectors

word_vectors = KeyedVectors.load('w2vec_models/complete.kv', mmap='r')

result = word_vectors.most_similar(positive=['mujer', 'rey'], negative=['hombre'])
print("{}: {:.4f}".format(*result[0]))

result = word_vectors.most_similar(positive=['tarde', 'comer'], negative=['pronto'])
print("{}: {:.4f}".format(*result[0]))

result = word_vectors.most_similar(positive=['parís', 'españa'], negative=['francia'])
print("{}: {:.4f}".format(*result[0]))

result = word_vectors.most_similar(positive=['nacer'])
print("{}: {:.4f}".format(*result[0]))

result = word_vectors.most_similar(positive=['nacer', 'juventud'])
print("{}: {:.4f}".format(*result[0]))

result = word_vectors.most_similar(positive=['nacer', 'aniversario'])
print("{}: {:.4f}".format(*result[0]))

result = word_vectors.most_similar(positive=['andar', 'caminar', 'rápido', 'prisa'], negative=['despacio'])
print("{}: {:.4f}".format(*result[0]))

print(word_vectors.doesnt_match("desayuno cereales cena almuerzo".split()))

print(word_vectors.doesnt_match("leche almuerzo cereales galletas".split()))

print(word_vectors.doesnt_match("leche cereales desayuno galletas".split()))

similarity = word_vectors.similarity('mesa', 'silla')
print(similarity)

result = word_vectors.similar_by_word("desagradecido", topn=100)
for r in result:
  print("{}: {:.4f}".format(*r), end=" / ")

frase_rajoy = 'Rajoy habló con los medios en Madrid'.lower().split()
frase_presidente = 'El presidente dio una rueda de prensa en la capital'.lower().split()
similarity = word_vectors.wmdistance(frase_rajoy, frase_presidente)
print("{:.4f}".format(similarity))

frase_1 = 'Pillaron a la famosa despistada con su nueva pareja'.lower().split()
frase_2 = 'Se espera que el tiempo de mañana sea soleado'.lower().split()
similarity = word_vectors.wmdistance(frase_1, frase_2)
print("{:.4f}".format(similarity))

frase_1 = 'Ayer tomé leche para desayunar'.lower().split()
frase_2 = 'Ayer desayuné leche'.lower().split()
similarity = word_vectors.wmdistance(frase_1, frase_2)
print("{:.4f}".format(similarity))

frase_1 = 'Ayer tomé leche para desayunar'.lower().split()
frase_2 = 'Ayer tomé leche para desayunar'.lower().split()
similarity = word_vectors.wmdistance(frase_1, frase_2)
print("{:.4f}".format(similarity))

distance = word_vectors.distance("comida", "comer")
print("{:.1f}".format(distance))

distance = word_vectors.distance("desayunar", "comer")
print("{:.1f}".format(distance))

distance = word_vectors.distance("comida", "perro")
print("{:.1f}".format(distance))

distance = word_vectors.distance("comida", "cascarrabias")
print("{:.1f}".format(distance))

sim = word_vectors.n_similarity(['sushi', 'tienda'], ['japonés', 'restaurante'])
print("{:.4f}".format(sim))

sim = word_vectors.n_similarity(['amarillo', 'azul', 'rojo', 'granate'], ['japonés', 'restaurante'])
print("{:.4f}".format(sim))

sim = word_vectors.n_similarity(['amarillo', 'azul', 'rojo', 'granate'], ['coche', 'auto'])
print("{:.4f}".format(sim))

sim = word_vectors.n_similarity(['comprar', 'vender', 'alquilar', 'prestar'], ['coche', 'auto'])
print("{:.4f}".format(sim))

sim = word_vectors.n_similarity(['comprar', 'vender', 'alquilar', 'prestar'], ['alquilar', 'prestar'])
print("{:.4f}".format(sim))

print(word_vectors.most_similar_to_given("escuchar", ["percibir", "oír", "entender", "hablar", "auriculares", "escucha"]))