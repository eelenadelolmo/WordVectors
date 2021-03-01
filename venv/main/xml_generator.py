import os
import re
import natsort
import shutil
from conllu import parse
import pyconll as pc
from ast import literal_eval

# Input dirs
dir_FN_annotated = 'framenet_annotated'
dir_TP_annotated = 'grew_annotated'

# Output dir for conllu
dir_output = 'out_fm'
shutil.rmtree(dir_output, ignore_errors=True)
os.makedirs(dir_output)

# Output dir for xml
dir_output_xml = 'out_xml'
shutil.rmtree(dir_output_xml, ignore_errors=True)
os.makedirs(dir_output_xml)


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

    return (found, modality_speaker)


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
        dict_ann = literal_eval(f.read())
        FN_annotated_list.append(dict_ann)

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

            sentence_str = txt_transformer_str(sentence.serialize())
            tokens_theme = forms_theme_rheme(sentence_main)[0].split()
            tokens_rheme = forms_theme_rheme(sentence_main)[1].split()
            pos_theme = pos_theme_rheme(sentence_main)[0]
            pos_rheme = pos_theme_rheme(sentence_main)[1]
            pos_rheme = pos_theme_rheme(sentence_main)[1]
            tokens_pos_theme = zip(tokens_theme, pos_theme)
            tokens_pos_rheme = zip(tokens_rheme, pos_rheme)

            verbs = get_main_verb_forms(sentence_main)

            xml_sentence += '\t\t<str>\n\t\t\t'
            xml_sentence += sentence_str
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
                frame = frame_head[0]

                # --> Tokens representing the head of the arguments
                frame_tokens = frame_head[1]

                # --> List of the ids (ordered) of the tokens corresponding to the head of the arguments
                h_ids = search_id(frame_tokens, sentence_main)

                for verb in verbs:
                    if verb in frame_tokens:
                        xml_sentence += '\t\t\t<main_frame type="' + frame + '" head="' + frame_tokens + '">'
                        final = '</main_frame>\n'

                    else:
                        xml_sentence += '\t\t\t<frame type="' + frame + '" head="' + frame_tokens + '">'
                        final = '</frame>\n'

                    for dep_ann in fm_anns[frame_head]:
                        # --> The type of argument
                        argument_type = dep_ann[0]

                        # --> The tokens representing the argument
                        argument_tokens = dep_ann[1]

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
