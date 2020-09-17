import torch
from transformers import BertForMaskedLM, BertTokenizer

# !wget https://users.dcc.uchile.cl/~jperez/beto/cased_2M/pytorch_weights.tar.gz
# !wget https://users.dcc.uchile.cl/~jperez/beto/cased_2M/vocab.txt
# !wget https://users.dcc.uchile.cl/~jperez/beto/cased_2M/config.json
# !tar -xzvf pytorch_weights.tar.gz
# !mv config.json pytorch/.
# !mv vocab.txt pytorch/.

tokenizer = BertTokenizer.from_pretrained("pytorch/", do_lower_case=True)
model = BertForMaskedLM.from_pretrained("pytorch/")
model.eval()

text = "[CLS] La [MASK] del problema nuclear actual es tirar bombas y se acabó porque no hay una respuesta clara. [SEP]"

tokens = tokenizer.tokenize(text)

masked_indxs = ()
id_masked = 0
for token in tokens:
    if token == "[MASK]":
        masked_indxs += (id_masked,)
    id_masked += 1

indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)

tokens_tensor = torch.tensor([indexed_tokens])

print(indexed_tokens)
#print(tokens_tensor)

predictions = model(tokens_tensor)[0]

for i,midx in enumerate(masked_indxs):
    idxs = torch.argsort(predictions[0,midx], descending=True)
    predicted_token = tokenizer.convert_ids_to_tokens(idxs[:25])
    print('MASK',i,':',predicted_token)

indexed_predicted_tokens = tokenizer.convert_tokens_to_ids(predicted_token)
print(indexed_predicted_tokens)
z