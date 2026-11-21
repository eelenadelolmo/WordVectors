# Normalización de las distancias y cálculo de los conjuntos referenciales

Código y ejemplos del módulo de identificación de los conjuntos referenciales (apartado 6.8.4 de la tesis), implementado en `venv/main/thematic_progression_annotator.py`.

## Normalización de las distancias WMD

La fórmula escogida para la normalización de las distancias obtenidas a partir de Word2vec ha sido la siguiente:

```python
norm = [1 - (float(i) / 20) for i in WMD_others]
```

Paralelamente, la fórmula de normalización de las métricas de similitud semántica de fastText ha sido la siguiente:

```python
norm = [1 - (float(i) / 3) for i in WMD_others]
```

La variable `WMD_others` contiene la lista de números que codifican las distancias de cada Tema (excepto del Tema con el que se está midiendo la distancia semántica, lo que explica que su nombre incluya others) según el algoritmo Word Movers Distance. Por tanto, la fórmula calcula una nueva lista de números en la que cada elemento se corresponde con la distancia original normalizada para estar incluida en el rango 0-1 para los valores que no sobrepasan el máximo observado, de manera que a mayores valores se les otorgan menores valores normalizados proporcionalmente y los valores que quedan por encima del máximo observado dan resultados negativos (que penalizan, por tanto, a los candidatos más lejanos).

## Ejemplos de normalización

A continuación presentamos ejemplos de normalización de las distancias para los distintos tipos de vectores:

```text
Word2vec embeddings
Theme: El Gobierno de Portugal.
Word Mover's distance from other themes:
El ministro : 14.597247929878712
El Ejecutivo portugués : 11.409844743225472
(norm.) El ministro : 0.2701376035060644
(norm.) El Ejecutivo portugués : 0.4295077628387264
```

```text
fastText embeddings
Theme: El Gobierno de Portugal.
Word Mover's distance from other themes:
El ministro : 1.920696549706698
El Ejecutivo portugués : 1.5724016325756325
(norm.) El ministro : 0.35976781676443403
(norm.) El Ejecutivo portugués : 0.4758661224747892
```

```text
Word2vec embeddings
Theme: La compañía, una de las más antiguas de Oriente Próximo.
Word Mover's distance from other themes:
Egypt Air , compañía estatal creada hace 67 años , : 8.192278182402802
eso : 23.582865094564525
El más grave y reciente : 12.621575807134041
Rayan , un ex general que dirige la empresa con mano de hierro , : 7.651253677676117
Su flota : 16.072189634575526
(norm.) Egypt Air , compañía estatal creada hace 67 años , : 0.5903860908798599
(norm.) eso : -0.17914325472822634
(norm.) El más grave y reciente : 0.3689212096432979
(norm.) Rayan , un ex general que dirige la empresa con mano de hierro , : 0.6174373161161941
(norm.) Su flota : 0.19639051827122367
```

```text
fastText embeddings
Theme: La compañía, una de las más antiguas de Oriente Próximo.
Word Mover's distance from other themes:
Egypt Air , compañía estatal creada hace 67 años , : 1.2547655056520217
eso : 3.0141134804875316
El más grave y reciente : 1.6601354385755371
Rayan , un ex general que dirige la empresa con mano de hierro , : 1.1559919641170269
Su flota : 2.467345720462398
(norm.) Egypt Air , compañía estatal creada hace 67 años , : 0.5817448314493261
(norm.) eso : -0.004704493495843787
(norm.) El más grave y reciente : 0.44662152047482095
(norm.) Rayan , un ex general que dirige la empresa con mano de hierro , : 0.6146693452943244
(norm.) Su flota : 0.17755142651253397
```

## Pesos de similitud entre Temas

El algoritmo de construcción de los conjuntos referenciales y los Temas que pertenecen a cada uno de ellos se basa en el cálculo ponderado que describimos arriba (cada Tema tiene asignado un peso según las métricas de distancia obtenidas a partir de los tres modelos de vectores preentrenados para español que hemos seleccionado). De acuerdo con ello, un ejemplo de  entrada del algoritmo para un texto con tres oraciones se correspondería con una estructura de datos como la siguiente:

```python
{
'El Gobierno de Portugal ':
{
'El Ejecutivo portugués': 1.7952949045179185,
'El ministro': 1.4921011879855186},
'El ministro ':
{
'El Ejecutivo portugués': 1.5473667826277597,
'El Gobierno de Portugal': 1.4921011879855182},
'El Ejecutivo portugués ':
{
'El Gobierno de Portugal': 1.7952949045179187,
'El ministro': 1.5473667826277597}
}
```

Como puede observarse, los pesos de las posibles correferencias de un Tema se almacenan en un diccionario cuyas claves son cada uno de los Temas de la oración en cuestión y cuyos valores son, de nuevo, de tipo diccionario, con el resto de los Temas de la oración como claves y sus pesos normalizados y ponderados como valores. El módulo permite establecer un valor crítico de similitud ponderada a partir del cual dos Temas se consideran correferentes.
