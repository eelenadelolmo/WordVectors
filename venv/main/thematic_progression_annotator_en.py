import os
import re
import ast
import html as html_import
import spacy
import shutil
import natsort
import secrets
import numpy as np
import pandas as pd
import pyconll as pc
import scipy.spatial
from conllu import parse
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from werkzeug.utils import secure_filename
from gensim.models.wrappers import FastText
from xml.etree.ElementTree import ElementTree
from sentence_transformers import SentenceTransformer
from flask import Flask, flash, request, redirect, send_file, render_template, url_for
import torch
import gc

# Pending: precedence of comparisons


torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)
del torch
gc.collect()


# Loading Spacy English model
model_sm_path = "Spacy_models/en_core_web_sm-2.3.1/en_core_web_sm/en_core_web_sm-2.3.1"
nlp = spacy.load(model_sm_path)


## XML transformer directories

# Input dirs
dir_FN_annotated = 'framenet_annotated'
dir_TP_annotated = 'grew_annotated'

# Output dir for conllu
dir_output = 'out_fm'

# Output dir for xml
dir_output_xml = 'out_xml_for_coref'

shutil.rmtree(dir_TP_annotated, ignore_errors=True)
os.makedirs(dir_TP_annotated + '/')

shutil.rmtree(dir_FN_annotated, ignore_errors=True)
os.makedirs(dir_FN_annotated + '/')


## TP annotator directories

# Directory containing the XML version of the annotated corpora
input_dir_coref = dir_output_xml

# Directory containing the temporal output in XML of the coreference annotation for whole themes and rhemes
output_dir_tmp = 'out_xml_tmp'

# Directory containing the output in XML of this module (coreference annotator for whole themes and rhematic mentions)
output_dir = 'out_xml'

# Directory containing the output in HTML of this module
output_dir_html = 'out_html'

# Directory containing the visual matrix output in png of this module
output_dir_plot = 'out_plot'


# Pretty-printing a dictionary with a n-tabs indentation
def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))


# Given a string composed of one or several tokens and a sentence
# Returns the ordered list of ids of the tokens in the sentence
def search_id(toks, sent):
    toks = re.sub('& quot ;', '&quot;', toks)
    toks_list = toks.split(' ')
    lines = sent.split('\n')
    ids = list()

    # Keeping the ids of every matched line for every token
    for token in toks_list:
        for line in lines:
            if len(line) > 1 and token == line.split('\t')[1]:
                ids.append(line.split('\t')[0])

    # When there are more matched ids than tokens to matchs
    if len(ids) > len(toks_list):

        # Creating a new list (by value, not reference) for not losing elements in the iteration for deleting
        ids_copy = ids[:]

        n_matches = 1
        n_searching = 0
        sig = ""

        ids_iter = iter(ids_copy)

        for x in ids_iter:

            # When all tokens have been correctly matched
            if n_matches == len(toks_list):
                # Keeping the matched tokens
                until = n_matches - 1
                ids = ids[:until]
                ids.append(sig)
                break

            # Calculating and searching the next id to match
            sig = str(int(x) + 1)

            # Deleting the current element if the next one is not in the rest of the list
            if sig not in ids[1:]:
                ids.remove(x)

            # If the next id to match is in the list
            else:
                # Keeping the elements until the current element and from the next matched id
                n_x = ids.index(x)
                n_sig = ids.index(sig)
                ids = ids[:n_x + 1] + ids[n_sig:]
                # Skipping the removed elements in the iteration
                n_to_skip = n_sig - n_x - 1
                for n in range(n_to_skip):
                    next(ids_iter)

                n_matches += 1

            if len(ids) == len(toks_list):
                break

            n_searching += 1

    return ids


# Given a list of features and a sentence
# Returns the subtree composed of the tokens of the sentence with a "yes" value in the features selected
def keep_annotations(anns, sent):
    lines = sent.split('\n')
    main = ""
    new_root = False

    # Solving the problem of sentences whose main clause does not include root
    for line in lines:
        line_fields = line.split('\t')

        # Getting the root when not in main theme or rheme
        if len(line_fields) >= 5 and "t=no" in line_fields[5] and "r=no" in line_fields[5]:
            new_root = True

    # When the original sentence root is not included in the compressed sentence
    if new_root:

        # Getting the conllu parsed version of the sentence
        try:
            sentences = parse(sent)
        except pc.exception.ParseError:
            sentences = parse(to_conllu(sent))
        sentence = sentences[0]

        # Getting the immediate sons of the head of the original root
        found = False
        hijos = [sentence.to_tree()]

        # While the head of the main clause hasn't been found
        while not found and len(hijos) > 0:
            for hijo in hijos:

                # Explore the tree until finding a t=yes or r=yes annotated node (the head of the main clause necessary)
                for ann in anns:
                    if hijo.token['feats'][ann] == 'yes':
                        hijo.token['deprel'] = 'root'
                        hijo.token['head'] = 0
                        found = True
                        break

            # Searching in the sons of every son in case immediate sons aren't the head of the selected main clause
            nietos = []
            for hijo in hijos:
                nietos.extend(hijo.children)
            hijos = nietos

        # Updating lines with the new root token annotated as such
        lines = sentence.serialize().split('\n')

    # Keeping only the lines corresponding to the main theme and rheme of the sentence
    for line in lines:
        line_fields = line.split('\t')

        if len(line_fields) >= 5 and ("t=yes" in line_fields[5] or "r=yes" in line_fields[5]):
            main += line + '\n'

    return main


# Given a sentence
# Returns a tuple with the tokens composing its theme and its rheme
def forms_theme_rheme(sent):
    lines = sent.split('\n')
    theme = ""
    rheme = ""

    for line in lines:
        line_fields = line.split('\t')
        if len(line_fields) >= 5 and ("t=yes" in line_fields[5]):
            theme += line_fields[1] + ' '
        if len(line_fields) >= 5 and ("r=yes" in line_fields[5]):
            rheme += line_fields[1] + ' '

    return (theme, rheme)


# Given a sentence
# Returns a tuple with the PoS of the tokens composing its theme and its rheme
def pos_theme_rheme(sent):
    lines = sent.split('\n')
    theme_pos_list = list()
    rheme_pos_list = list()

    for line in lines:
        line_fields = line.split('\t')
        if len(line_fields) >= 5 and ("t=yes" in line_fields[5]):
            theme_pos_list.append(line_fields[3])
        if len(line_fields) >= 5 and ("r=yes" in line_fields[5]):
            rheme_pos_list.append(line_fields[3])

    return (theme_pos_list, rheme_pos_list)


# Given a sentence
# Returns the list of the forms of the verbs contained in the main proposition
def get_main_verb_forms(sent):
    lines = sent.split('\n')
    main_verb_forms = list()

    for line in lines:
        line_fields = line.split('\t')
        if len(line_fields) >= 5 and ("m=yes" in line_fields[5]) and (line_fields[3][0] == 'v'):
            main_verb_forms.append(line_fields[1])
    return main_verb_forms


# Given a sentence
# Returns a tuple with a boolean True value if a reported speech subject if annotated and its string; or a boolean False value and an empty string otherwise
def get_modalitySpeaker(sent):
    lines = sent.split('\n')
    modality_speaker = ""
    found = False

    for line in lines:
        line_fields = line.split('\t')
        if len(line_fields) >= 5 and ("rep=yes" in line_fields[5]):
            found = True
            modality_speaker += line_fields[1] + ' '

    return (found, modality_speaker.replace('&', '').replace('<', '').replace('>', ''))


# Given a sentence
# Returns a tuple with the lists with the ids of the elements of its theme and rheme
def ids_theme_rheme(sent):
    lines = sent.split('\n')
    theme = list()
    rheme = list()

    for line in lines:
        line_fields = line.split('\t')
        if len(line_fields) >= 5 and ("t=yes" in line_fields[5]):
            theme.append(line_fields[0])
        if len(line_fields) >= 5 and ("r=yes" in line_fields[5]):
            rheme.append(line_fields[0])
    return (theme, rheme)


# Reindexing the tokens of a sentence to avoid gaps in order to match continuous annotations along them
def reset_ids(sent):
    sent_lines = sent.split('\n')

    # Beggining ids from 1
    id_new = 1

    # Creating a tmp file with both new a original ids
    tmp = ""

    # Keeping previous and new ids with the format new_id-prev_id
    for line in sent_lines:
        if len(line) > 1:
            tmp += str(id_new) + '-' + line + '\n'
            id_new += 1

    # Creating the file for the resetted ids
    resetted = sent

    # Getting the original id
    for line_tmp in tmp.split('\n'):
        if len(line_tmp) > 1:
            id_old = re.search('-(.+)', line_tmp.split('\t')[0]).group(1)
            id_new = re.search('(.+)-', line_tmp.split('\t')[0]).group(1)

            # Resetted is rewritten for every pair id_old / id_new
            lines_resetted = resetted.split('\n')
            resetted = ""

            for line_resetted in lines_resetted:
                if len(line_resetted) > 1 and line_resetted.split('\t')[6] == id_old:
                    line_modified = re.sub(id_old + r'(.+?)', str(id_new) + r'\1', line_resetted)
                elif len(line_resetted) > 1 and line_resetted.split('\t')[0] == id_old:
                    line_modified = re.sub(r'^' + id_old + '(.+?)', str(id_new) + r'\1', line_resetted)
                else:
                    line_modified = line_resetted

                if (len(line_modified) > 1):
                    resetted += line_modified + '\n'

    return resetted


def to_conllu(sentence):
    lines = sentence.split('\n')
    ok = ""
    for l in lines:
        if len(l) > 1:
            line_ok = l[:-1] + '\t_\t_\n'
            ok += line_ok
        else:
            ok += (l + '\n')
    return ok


# Transforms a conllu sentence into the string with its forms
def txt_transformer(file_conllu):
    s_list = list()
    with open(file_conllu, 'r') as f:
        conll = pc.load_from_string(f.read())
    for s in conll:
        s_txt = ""
        for word in s[:-1]:
            s_txt = s_txt + " " + word.form
        s_txt = s_txt + ".\n"
        s_list.append(s_txt)
    return s_list


# Transforms a conllu sentence into the string with its forms
def txt_transformer_str(sentence):
    try:
        conll = pc.load_from_string(sentence)
    except pc.exception.ParseError:
        conll = pc.load_from_string(to_conllu(sentence))
    sent = conll[0]
    s_txt = ""
    for word in sent:
        s_txt = s_txt + " " + word.form
    # s_txt = s_txt + ".\n"
    s_txt = re.sub(r" ([.,:?!])", r"\1", s_txt)
    s_txt = re.sub(r"([¿]) ", r"\1", s_txt)
    s_txt = re.sub(r"&quot; (.+?) &quot;", r'"\1"', s_txt)
    return s_txt


# Returns a file name without the extension
def remove_ext(filename):
    return filename[:filename.rfind(".")]


def make_archive(source, destination):
        base = os.path.basename(destination)
        name = base.split('.')[0]
        format = base.split('.')[1]
        archive_from = os.path.dirname(source)
        archive_to = os.path.basename(source.strip(os.sep))
        shutil.make_archive(name, format, archive_from, archive_to)
        shutil.move('%s.%s'%(name,format), destination)


DOWNLOAD_FOLDER = 'downloads'
UPLOAD_FOLDER_mapper = 'uploads'
DOWNLOAD_FOLDER_mapper = 'downloads_mapper/out'

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER_mapper'] = UPLOAD_FOLDER_mapper
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.after_request
def add_header(response):
  response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
  if ('Cache-Control' not in response.headers):
    response.headers['Cache-Control'] = 'public, max-age=600'
  return response

@app.route('/upload-tp-ann')
def upload_form():
    return render_template('upload_thematic_progression.html')

@app.route('/upload-tp-ann', methods=['POST'])
def upload_file():
    if request.method == 'POST':

        shutil.rmtree(dir_output, ignore_errors=True)
        os.makedirs(dir_output)
        shutil.rmtree(dir_output_xml, ignore_errors=True)
        os.makedirs(dir_output_xml)
        shutil.rmtree(output_dir_tmp, ignore_errors=True)
        os.makedirs(output_dir_tmp)
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir)
        shutil.rmtree(output_dir_html, ignore_errors=True)
        os.makedirs(output_dir_html)
        shutil.rmtree(output_dir_plot, ignore_errors=True)
        os.makedirs(output_dir_plot)

        # check if the post request has the file part
        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist('files[]')
        # if user does not select file, browser also
        # submit an empty part without filename
        for file in files:
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            # if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_conll):
            filename = secure_filename(file.filename)
            if not filename.endswith('.conllu'):
                file.save(os.path.join(dir_FN_annotated, filename))
            if filename.endswith('.conllu'):
                file.save(os.path.join(dir_TP_annotated, filename))
            file.stream.seek(0)

    return render_template('upload_thematic_progression.html')


@app.route('/TP_annotate', methods=['POST'])
def TP_annotate():

    import torch
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)
    del torch
    gc.collect()

    ## Pretrained models

    # Loading BERT model for English
    # np.set_printoptions(threshold=100)
    # BETO_model = SentenceTransformer('BERT_ROBERTA_model')

    # Loading Word2vec model for English
    # w2vec_models = KeyedVectors.load_word2vec_format('w2vec_models_en/GoogleNews-vectors-negative300.bin', binary=True)

    # Loading FastText model for English
    FastText_models = FastText.load_fasttext_format('FastText_models/cc.en.300.bin')



    # List of FrameNet annotations (one list element per sentence)
    FN_annotated_list = list()

    ## Data structure:
    # - Every FrameNet annotation consist of a Python dictionary with
    # Key: a Python tuple (type of frame, head string)
    # Value: list of arguments
    # - Every argument is a Python tuple (type of FrameNet relation, string)

    ## Creating a XML file with the theme/rheme and the FrameNet annotations information

    # XML content to save for the text
    xml = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>

    <!DOCTYPE text [
    	<!ELEMENT text (sentence+)>
    	    <!ATTLIST text id CDATA #REQUIRED>
    	<!ELEMENT sentence (str, theme, rheme, semantic_roles)>
    		<!ELEMENT str (#PCDATA)>
    		<!ELEMENT theme (token*)>
    		<!ELEMENT rheme (token*)>
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

    # Getting the ordered list with the FrameNet annotations for every sentence
    files_fm = natsort.natsorted(os.listdir(dir_FN_annotated))
    for file in files_fm:
        with open(dir_FN_annotated + '/' + file, 'r') as f:

            texto = f.read()
            f.seek(0)
            SRL = dict()

            anotaciones = texto.split('\n\n')
            for anotacion in anotaciones:
                lineas = anotacion.split('\n')

                # Getting the frame identificator of the anotacion and the corresponding string, which may be after the first labelled role
                for linea in lineas:
                    if len(linea) > 1:
                        frame_id = linea.split('\t')[-3]
                        frame_type = linea.split('\t')[-2]
                        frame_str = linea.split('\t')[1]
                        if frame_id != '_':
                            anotacion_frame = frame_type
                            SRL[(anotacion_frame, frame_str)] = list()
                            break

                n_linea = 0
                for linea in lineas:
                    if len(linea) > 1:
                        n_linea += 1
                        iob = linea.split('\t')[-1]
                        SRL_tag = iob[2:]
                        string = linea.split('\t')[1]
                        if 'S-' in iob and linea.split('\t')[-3] == '_':
                            SRL[(anotacion_frame, frame_str)].append((SRL_tag, string))
                        if 'B-' in iob:
                            ann_tmp = string
                            lineas_siguientes = lineas[n_linea:]

                            for linea_siguiente in lineas_siguientes:
                                iob_sig = linea_siguiente.split('\t')[-1]
                                str_sig = linea_siguiente.split('\t')[1]
                                if 'I-' in iob_sig:
                                    ann_tmp = ann_tmp + " " + str_sig
                                else:
                                    SRL[(anotacion_frame, frame_str)].append((SRL_tag, ann_tmp))
                                    break

            FN_annotated_list.append(SRL)

    n_sentence = 0
    files_grew = natsort.natsorted(os.listdir(dir_TP_annotated))
    for file_grew in files_grew:
        with open(dir_TP_annotated + '/' + file_grew) as f:
            sentences = parse(f.read())
            xml_sentence = xml + '<text id="' + remove_ext(file_grew) + '">\n\n\n'

            # --> String representation of the conllu structure of the sentences in a text
            fm_annotated = ""

            for sentence in sentences:

                # Keeping the reporter modality marker before transforming the original sentence
                reporter = get_modalitySpeaker(sentence.serialize())

                # Reducing every sentence to the theme + rheme of the main clause
                sentence_main = keep_annotations(['t', 'r'], sentence.serialize())

                # Cleaning repeated empty lines in the sentence
                while True:
                    sent_clean = re.sub('\n\n', '\n', sentence_main)
                    if sent_clean == sentence_main:
                        break
                    sentence_main = sent_clean

                # Rewriting the ids: beginning from 1 and with no gaps
                sentence_main = reset_ids(sentence_main)

                # Cleaning repeated empty lines in the sentence
                while True:
                    sent_clean = re.sub('\n\n', '\n', sentence_main)
                    if sent_clean == sentence_main:
                        break
                    sentence_main = sent_clean

                fm_anns = FN_annotated_list[n_sentence]
                n_sentence += 1

                # print('________________________________________________________')
                # print(sentence_main)

                xml_sentence += '\t<sentence>\n'

                sent_str = txt_transformer_str(sentence.serialize().replace('&', '').replace('<', '').replace('>', ''))
                tokens_theme = forms_theme_rheme(sentence_main)[0].replace('&', '').replace('<', '').replace('>', '').split()
                tokens_rheme = forms_theme_rheme(sentence_main)[1].replace('&', '').replace('<', '').replace('>', '').split()
                pos_theme = pos_theme_rheme(sentence_main)[0]
                pos_rheme = pos_theme_rheme(sentence_main)[1]
                pos_rheme = pos_theme_rheme(sentence_main)[1]
                tokens_pos_theme = zip(tokens_theme, pos_theme)
                tokens_pos_rheme = zip(tokens_rheme, pos_rheme)

                verbs = get_main_verb_forms(sentence_main)

                xml_sentence += '\t\t<str>\n\t\t\t'
                xml_sentence += sent_str
                xml_sentence += '\n\t\t</str>\n'

                xml_sentence += '\t\t<theme>\n\t\t\t'
                for token_pos in tokens_pos_theme:
                    xml_sentence += '<token pos="' + token_pos[1] + '">' + token_pos[0] + '</token>'
                xml_sentence += '\n\t\t</theme>\n'

                xml_sentence += '\t\t<rheme>\n\t\t\t'
                for token_pos in tokens_pos_rheme:
                    xml_sentence += '<token pos="' + token_pos[1] + '">' + token_pos[0] + '</token>'
                xml_sentence += '\n\t\t</rheme>\n'

                xml_sentence += '\t\t<semantic_roles>\n'

                if reporter[0]:
                    xml_sentence += '\t\t\t<frame type="Modality_Reporter" head="' + reporter[1] + '"></frame>\n'

                # Getting the ids of the tokens conforming the head of a FrameNet frame
                for frame_head in fm_anns:

                    # --> FrameNet frame
                    frame = frame_head[0].replace('&', '').replace('<', '').replace('>', '')

                    # --> Tokens representing the head of the arguments
                    frame_tokens = frame_head[1].replace('&', '').replace('<', '').replace('>', '')

                    # --> List of the ids (ordered) of the tokens corresponding to the head of the arguments
                    h_ids = search_id(frame_tokens, sentence_main)

                    main_frame = False
                    for verb in verbs:
                        if verb in frame_tokens:
                            xml_sentence += '\t\t\t<main_frame type="' + frame + '" head="' + frame_tokens + '">'
                            final = '</main_frame>\n'
                            main_frame = True
                            break

                    if not main_frame:
                        xml_sentence += '\t\t\t<frame type="' + frame + '" head="' + frame_tokens + '">'
                        final = '</frame>\n'

                    for dep_ann in fm_anns[frame_head]:
                        # --> The type of argument
                        argument_type = dep_ann[0].replace('&', '').replace('<', '').replace('>', '')

                        # --> The tokens representing the argument
                        argument_tokens = dep_ann[1].replace('&', '').replace('<', '').replace('>', '')

                        # --> List of the ids (ordered) of the tokens correponding to the argument
                        h_dep_ids = search_id(argument_tokens, sentence_main)
                        # print(argument_tokens, h_dep_ids, '\n' + sentence_main)

                        xml_sentence += '\n\t\t\t\t<argument type="' + argument_type + '" dependent="' + argument_tokens + '"/>'
                    xml_sentence += final
                xml_sentence += '\t\t</semantic_roles>\n'
                xml_sentence += '\t</sentence>\n\n\n'

                fm_annotated += (sentence_main + '\n')

            # Creating a new file with the selected subtree with the FrameNet annotations
            with open(dir_output + '/' + file_grew, "w") as h:
                lines = fm_annotated.split('\n')
                ok = ""
                for l in lines:
                    if len(l) > 1:
                        line_ok = l[:-1] + '\t_\t_\n'
                        ok += line_ok
                    else:
                        ok += (l + '\n')
                h.write(ok)
                h.close()

            xml_sentence += '</text>'

            with open(dir_output_xml + '/' + remove_ext(file_grew) + '.xml', "w") as xml_file:
                xml_file.write(xml_sentence)
                xml_file.close()

    ##____________________________________________________________________________________________________________
    ##____________________________________________________________________________________________________________
    ##____________________________________________________________________________________________________________
    ## Getting the themes, rhemes and semantic arguments of the rhemes

    for file_xml in os.listdir(input_dir_coref):

        with open(input_dir_coref + '/' + file_xml) as archivo:
            texto = archivo.read().encode(encoding='utf-8').decode('utf-8')
            texto = re.sub('\\& ', '', texto)

        with open(output_dir_tmp + '/' + file_xml, 'w') as nuevo:
            nuevo.write(texto)
            nuevo.close()

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
                        # Only if both theme and rheme have been annotated
                        if len(theme) == 0:
                            rheme = ''
                        if len(theme) > 0:
                            for token in child:
                                rheme += token.text.strip() + ' '
                            # Only if both theme and rheme have been annotated
                            if len(rheme) == 0:
                                theme = ''
                    if child.tag == 'semantic_roles':
                        for frame in child:
                            if frame.tag == 'main_frame':
                                for arg in frame:
                                    # In order not to get concepts not in the rheme
                                    # print(arg.attrib['dependent'])
                                    # print(rheme)
                                    if arg.attrib['dependent'] in rheme:
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

            """ OOM
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
                    themes_ranking[theme.strip()][t_ord[idx].strip()] = 1 - distance

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
            """


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

                themes_ranking[theme.strip()] = dict()

                for other in others:
                    WMD = FastText_models.wv.wmdistance(theme, other)
                    WMD_others.append(WMD)
                    # print("-", other + ":", WMD)
                    themes_ranking[theme.strip()][other.strip()] = WMD

                """ OOM
                # Normalizing the Word Movers Distance value to a 0-1 range
                # norm = [1 - (float(i) / sum(WMD_others)) for i in WMD_others]
                norm = [1 - (float(i) / 3) for i in WMD_others]
                for n, other in enumerate(others):
                    # print("- (norm.)", other + ":", norm[n])
                    themes_ranking[theme.strip()][other.strip()] += norm[n]
                """

                # print('\n\n')

            ##____________________________________________________________________________________________________________
            ##____________________________________________________________________________________________________________
            ##____________________________________________________________________________________________________________
            ## Getting rheme noun phrases-theme similarity measures

            # print('\n\n\n\n========================================================')
            # print("** Matching rhemes to themes **")
            rhemes_themes_ranking = dict()

            """ oom
            # print("\nSentence transformers with BETO as a model")

            rheme_embeddings = BETO_model.encode(r_ord_noun_phrases_all)

            # Find the closest closest_n sentences of the corpus for each query sentence based on cosine similarity
            closest_n = len(theme_embeddings)
            # Changed!!!!
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
                    rhemes_themes_ranking[rheme.strip()][t_ord[idx].strip()] = 1 - distance

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
            """

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

                rhemes_themes_ranking[rheme.strip()] = dict()

                for other in others:
                    WMD = FastText_models.wv.wmdistance(rheme, other)
                    WMD_others.append(WMD)
                    # print("-", other + ":", WMD)
                    rhemes_themes_ranking[rheme.strip()][other.strip()] = WMD

                """ OOM
                # Normalizing the Word Movers Distance value to a 0-1 range
                # norm = [1 - (float(i) / sum(WMD_others)) for i in WMD_others]
                norm = [1 - (float(i) / 3) for i in WMD_others]
                for n, other in enumerate(others):
                    # print("- (norm.)", other + ":", norm[n])
                    rhemes_themes_ranking[rheme.strip()][other.strip()] += norm[n]
                """

                # print('\n\n')

            ##____________________________________________________________________________________________________________
            ##____________________________________________________________________________________________________________
            ##____________________________________________________________________________________________________________
            ## Getting rheme main semantic frame arguments-rheme main semantic frame arguments similarity measures

            # print('\n\n\n\n========================================================')
            # print("** Matching rhematic main frame arguments to rhematic main frame arguments **")

            rhemes_sr_ranking = dict()

            """ OOM
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
                    rhemes_sr_ranking[rheme.strip()][r_ord_sem_roles_all[idx].strip()] = 1 - distance

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
            """

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

                """ OOM
                # Normalizing the Word Movers Distance value to a 0-1 range
                # norm = [1 - (float(i) / sum(WMD_others)) for i in WMD_others]
                norm = [1 - (float(i) / 3) for i in WMD_others]
                for n, other in enumerate(others):
                    # print("- (norm.)", other + ":", norm[n])
                    rhemes_sr_ranking[rheme.strip()][other.strip()] += norm[n]
                """

                # print('\n\n')

            ##____________________________________________________________________________________________________________
            ##____________________________________________________________________________________________________________
            ##____________________________________________________________________________________________________________
            ## Getting rheme noun phrases-rheme noun phrases similarity measures

            # print('\n\n\n\n========================================================')
            # ("** Matching rheme noun phrases to rheme noun phrases **")

            rhemes_ranking = dict()

            """ OOM
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
                    rhemes_ranking[rheme.strip()][r_ord_noun_phrases_all[idx].strip()] = 1 - distance

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
            """

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

                rhemes_ranking[rheme.strip()] = dict()

                for other in others:
                    WMD = FastText_models.wv.wmdistance(rheme, other)
                    WMD_others.append(WMD)
                    # print("-", other + ":", WMD)
                    rhemes_ranking[rheme.strip()][other.strip()] = WMD

                """ OOM
                # Normalizing the Word Movers Distance value to a 0-1 range
                # norm = [1 - (float(i) / sum(WMD_others)) for i in WMD_others]
                norm = [1 - (float(i) / 3) for i in WMD_others]
                for n, other in enumerate(others):
                    # print("- (norm.)", other + ":", norm[n])
                    rhemes_ranking[rheme.strip()][other.strip()] += norm[n]
                """

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
                        if rhemes_ranking[r][r_c] > threshold_rhemes_np and r not in [r_t_coref for r_t_coref, id_coref
                                                                                      in rheme_theme_id]:
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
                                regex = r'</token><token pos="[^>]+?">'.join(
                                    e[0].strip().replace("(", "\(").replace(")", "\)").split())
                                regex = r'<token pos="[^>]+?">' + regex + '</token>'
                                encontrados_start_end.append([(m.start(0), m.end(0)) for m in
                                                              re.finditer(regex, ET.tostring(child, encoding="unicode"),
                                                                          re.DOTALL)])
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
        <!ELEMENT sentence (str, theme, rheme, semantic_roles)>
            <!ELEMENT str (#PCDATA)>
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

                out.write(html_import.unescape(xml_new))

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
        <!ELEMENT sentence (str, theme, rheme, semantic_roles)>
            <!ELEMENT str (#PCDATA)>
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
                xml_nue += '<str>\t\t\t'
                xml_nue += re.search('<str>(.+?)</str>', sentence_str, re.DOTALL).group(1)
                xml_nue += '</str>\n\t\t'
                match = re.search('<theme.+?>(.+?)</theme>', sentence_str, re.DOTALL)
                if (match):
                    theme_str = match.group(0)
                else:
                    theme_str = '<theme>\n\t\t</theme>'

                xml_nue += theme_str + '\n\t\t'

                rheme_str = re.search('</theme>(.+?)<to_annotate>.+?</to_annotate>', sentence_str, re.DOTALL).group(
                    1).strip()
                # Pending to solve
                # to_ann_list = [x for x in ast.literal_eval(re.search('<to_annotate>(.+?)</to_annotate>', xml_ant).group(1))]
                to_ann_list = [x for x in
                               ast.literal_eval(re.search('<to_annotate>(.+?)</to_annotate>', sentence_str).group(1)) if
                               len(x) == 3]

                extra_len = 0

                to_ann_list.sort()
                lista_rangos = [el[0] for el in to_ann_list]

                fin_old = 0

                for tup in to_ann_list:

                    # Avoding multiclass mentions
                    if lista_rangos.count(tup[0]) == 1:

                        prin = tup[0][0] + extra_len
                        fin = tup[0][1] + extra_len

                        # Avoiding nesting and overlapping
                        if prin >= fin_old and '<m' not in rheme_str[prin:fin] and '</m' not in rheme_str[prin:fin] and \
                                rheme_str[fin - 1] != '<':
                            start_tag = '<mention concept_ref="' + tup[2] + '">'
                            end_tag = '</mention>'
                            extra_len = extra_len + len(start_tag + end_tag)
                            rheme_str = rheme_str[:prin] + start_tag + rheme_str[prin:fin] + end_tag + rheme_str[fin:]
                            fin_old = fin

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

    colors = ['708090', 'D2691E', '556B2F', 'FF3300', '000080', '2F4F4F', '4B0082', '8B4513', '6A5ACD', '663399',
              'BDB76B', 'FFD700', '00FF7F', 'D2B48C', 'DEB887', '32CD32', 'FFFACD', '0000CD', '008000', 'FFFAF0',
              '6B8E23', '90EE90', '7B68EE', 'FFFFF0', '5F9EA0', 'FFFFFF', '6495ED', '00CED1', '808080', '00008B',
              'FFF8DC', '4169E1', 'FF1493', 'FF6347', 'F4A460', '7FFF00', '808000', 'F5F5DC', '8B008B', 'FFF0F5',
              'F0FFF0', '9ACD32', 'ADFF2F', 'FF7F50', 'DC143C', 'FF69B4', 'FFFFE0', 'ADD8E6', 'FFEFD5', '8A2BE2',
              'DAA520', '7FFFD4', 'E0FFFF', 'BA55D3', 'FF8C00', '20B2AA', 'AFEEEE', 'B22222', '008080', '2E8B57',
              'CD853F', 'B0C4DE', 'FAEBD7', '000000', '228B22', '008B8B', '006400', '8FBC8F', '778899', 'FFDAB9',
              'FFFAFA', '696969', 'FFE4B5', 'E9967A', 'F0FFFF', '66CDAA', '800080', '87CEEB', 'D3D3D3', 'C0C0C0',
              'FA8072', '4682B4', 'F5F5F5', 'DCDCDC', '00FFFF', '48D1CC', 'B0E0E6', 'FFA07A', 'FF0000', '00FA9A',
              'A9A9A9', 'FF4500', 'DDA0DD', 'E6E6FA', 'FFEBCD', 'BC8F8F', 'EE82EE', 'FFA500', 'A0522D', '8B0000',
              'F8F8FF', 'FDF5E6', '98FB98', '9370DB', '191970', 'FFF5EE', 'FF00FF', 'EEE8AA', 'FAFAD2', '800000',
              'FFC0CB', '9932CC', 'B8860B', '00BFFF', 'FFDEAD', 'FFB6C1', 'DB7093', '00FF00', '40E0D0', 'F5DEB3',
              'FFFF00', 'FAF0E6', '3CB371', 'D8BFD8', '9400D3', 'C71585', 'DA70D6', 'F0E68C', '0000FF', 'FFE4E1',
              'F5FFFA', 'CD5C5C', '483D8B', '87CEFA', '8B4513', '7CFC00', 'F0F8FF', 'A52A2A', '1E90FF', 'F08080']

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

            <h1>Texto</h1>

        """

        with open(output_dir + '/' + file_xml) as xml:

            texto = xml.read()
            root = ET.fromstring(texto)

            for sentence in root.iter('sentence'):
                for child in sentence:
                    if child.tag == 'str':
                        html += child.text + ' '

            ids_colors = dict()
            n_concepts = 0

            html += '''

            <h1>Conceptos</h1>'''

            for concepts in root.iter('concepts'):
                for concept in concepts:
                    ids_colors[concept.attrib['id']] = colors[n_concepts % 140]
                    html += '<p style="color:#' + str(colors[n_concepts % 140]) + ';">' + concept.text.strip() + ' (' + \
                            concept.attrib['id'] + ')</p>'
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

                html += '<p> [ ' + theme_id_str + ' / ' + rheme_id_str + ' ] <span style="white-space:nowrap;color:#' + str(
                    theme_color) + ';">' + theme + '</span> ////<br/>' + '<nobr>' + rheme + '</nobr>'

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

    for_zip = 'downloads/out'
    shutil.rmtree(for_zip, ignore_errors=True)
    os.makedirs(for_zip)
    for_zip_tp_xml = 'downloads/out/xml'
    for_zip_tp_html = 'downloads/out/html'
    for_zip_tp_png = 'downloads/out/png'
    shutil.copytree(output_dir, for_zip_tp_xml)
    shutil.copytree(output_dir_html, for_zip_tp_html)
    shutil.copytree(output_dir_plot, for_zip_tp_png)

    shutil.rmtree(dir_TP_annotated, ignore_errors=True)
    os.makedirs(dir_TP_annotated + '/')
    shutil.rmtree(dir_FN_annotated, ignore_errors=True)
    os.makedirs(dir_FN_annotated + '/')

    make_archive(for_zip, DOWNLOAD_FOLDER + '/' + 'TP_annotated.zip')
    return redirect(url_for('download_file_tp_ann', filename='TP_annotated.zip'))


@app.route("/downloadfile-tp-ann/<filename>", methods = ['GET'])
def download_file_tp_ann(filename):
    return render_template('download_thematic_progression.html', value=filename)

@app.route('/return-files-tp-ann/<filename>', methods = ['GET'])
def return_files(filename):
    file_path = DOWNLOAD_FOLDER + '/TP_annotated.zip'
    return send_file(file_path, as_attachment=True, cache_timeout=0)




if __name__ == "__main__":
    secret = secrets.token_urlsafe(32)
    app.secret_key = secret
    app.run(host='0.0.0.0', port="5003")

