import io
import shutil
import os
import re
from googletrans import Translator
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, send_file, render_template, url_for
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
import nltk
import scipy
import pprint
import secrets
import pyconll as pc
nltk.download('punkt')


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
# Takes a conllu file as input and returs a str with one sentence per line
def txt_transformer(file_conllu):
    s_list = list()
    with open(file_conllu, 'r') as f:
        ok = f.read()
    try:
        conll = pc.load_from_string(ok)
    except pc.exception.ParseError:
        conll = pc.load_from_string(to_conllu(ok))
    for s in conll:
        s_txt = ""
        for word in s[:-1]:
            s_txt = s_txt + " " + word.form
        s_txt = s_txt.strip() + ".\n"
        s_list.append(s_txt)
    return u''.join(s_list).encode('utf-8')


def make_archive(source, destination):
        base = os.path.basename(destination)
        name = base.split('.')[0]
        format = base.split('.')[1]
        archive_from = os.path.dirname(source)
        archive_to = os.path.basename(source.strip(os.sep))
        shutil.make_archive(name, format, archive_from, archive_to)
        shutil.move('%s.%s'%(name,format), destination)


UPLOAD_FOLDER = '/home/elena/PycharmProjects/WordVectors/venv/main/uploads'
DOWNLOAD_FOLDER = '/home/elena/PycharmProjects/WordVectors/venv/main/downloads'

shutil.rmtree(UPLOAD_FOLDER + '/', ignore_errors=True)
shutil.rmtree(DOWNLOAD_FOLDER + '/', ignore_errors=True)
os.makedirs(UPLOAD_FOLDER + '/')
os.makedirs(DOWNLOAD_FOLDER + '/')

ALLOWED_EXTENSIONS_txt = {'txt'}
ALLOWED_EXTENSIONS_conll = {'conll'}
ALLOWED_EXTENSIONS_conllu = {'conllu'}

def allowed_file(filename, extension):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in extension

app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


@app.after_request
def add_header(response):
  response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
  if ('Cache-Control' not in response.headers):
    response.headers['Cache-Control'] = 'public, max-age=600'
  return response

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':

        # check if the post request has the file part
        # if 'file' not in request.files:
        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)

        # file = request.files['file']
        files = request.files.getlist('files[]')

        new_dir = UPLOAD_FOLDER + '/salida'
        shutil.rmtree(new_dir, ignore_errors=True)
        os.makedirs(new_dir)

        for file in files:
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_txt):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                file.stream.seek(0)

            elif file and allowed_file(file.filename, ALLOWED_EXTENSIONS_conllu):
                filename = secure_filename(file.filename)[:-7] + '.txt'
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                text_sentences = txt_transformer(UPLOAD_FOLDER + '/' + filename)
                with io.open(UPLOAD_FOLDER + '/' + filename, 'w', encoding="utf-8") as f:
                    f.write(text_sentences.decode('utf-8'))
                    f.seek(0)
                    f.close()

            else:
                print("Formato de archivo no válido.")

            new_dir_sent = UPLOAD_FOLDER + '/salida/en_' + filename[:-4]
            shutil.rmtree(new_dir_sent, ignore_errors=True)
            os.makedirs(new_dir_sent)

            with io.open(UPLOAD_FOLDER + '/' + filename, 'r', encoding='utf8') as f:
                lines = f.readlines()
                n_lines = 0
                translator = Translator()
                for line in lines:
                    n_lines += 1
                    with io.open(new_dir_sent + '/en_' + filename[:-4] + '_' + str(n_lines) + '.txt', 'w', encoding='utf8') as f_new_1:
                        translation = translator.translate(text=re.sub('&quot;', '"', line), src='es', dest='en')
                        f_new_1.write(translation.text)
                        f_new_1.close()
                    # Saving original files too
                    with io.open(new_dir_sent + '/' + filename[:-4] + '_' + str(n_lines) + '.txt', 'w', encoding='utf8') as f_new_2:
                        f_new_2.write(line)
                        f_new_2.close()

        make_archive(new_dir, DOWNLOAD_FOLDER + '/en_salida' + '.zip')

        return redirect(url_for('download_file', filename='salida.zip'))

    return render_template('main.html')


@app.route('/upload-frame-ann-en')
def upload_form():
    shutil.rmtree(UPLOAD_FOLDER + '/', ignore_errors=True)
    os.makedirs(UPLOAD_FOLDER + '/')
    return render_template('upload_frame_ann_en.html')


@app.route('/upload-frame-ann-en', methods=['POST'])
def upload_file():
    if request.method == 'POST':

        shutil.rmtree(UPLOAD_FOLDER + '/', ignore_errors=True)
        shutil.rmtree(DOWNLOAD_FOLDER + '/', ignore_errors=True)
        os.makedirs(UPLOAD_FOLDER + '/')
        os.makedirs(DOWNLOAD_FOLDER + '/')

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
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.stream.seek(0)

    return render_template('upload_frame_ann_en.html')


@app.route('/mapping_SRL', methods=['POST'])
def mappingsrl():

    shutil.rmtree(DOWNLOAD_FOLDER + '/', ignore_errors=True)
    os.makedirs(DOWNLOAD_FOLDER + '/')

    model = SentenceTransformer('distiluse-base-multilingual-cased')

    files_ann = list()
    files_es = list()

    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.split('.')[-1] == 'conll':
            files_ann.append(filename)
        else:
            files_es.append(filename)

    for name in files_ann:
        with io.open(UPLOAD_FOLDER + "/" + name, 'r', encoding='utf8') as f:
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

        # Creating the embeddings for all the subphrases of the Spanish sentences
        for original_filename in files_es:
            if original_filename + '.conll' == name:
                with io.open(UPLOAD_FOLDER + "/" + original_filename, 'r', encoding='utf8') as f:
                    sentence_original = f.read()
                    sentence_original_tokens = word_tokenize(sentence_original)
                    sentences_original_all_subphrases = [sentence_original_tokens[i: j] for i in
                                         range(len(sentence_original_tokens))
                                         for j in range(i + 1, len(sentence_original_tokens) + 1)]

        subfrases = list()
        for s in sentences_original_all_subphrases:
            subfrases.append(' '.join(s))

        sentence_embeddings = model.encode(subfrases)

        # Creating the dictionary for the Spanish annotations
        SRL_es = dict()

        for frame in SRL:
            query_frame_type = frame[0]
            query_frame_str = [frame[1]]
            query_frame_str_embedding = model.encode(query_frame_str)
            closest_n = 1
            distances = scipy.spatial.distance.cdist([query_frame_str_embedding[0]], sentence_embeddings, "cosine")[0]
            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])
            for idx, distance in results[0:closest_n]:
                query_frame_str_es = subfrases[idx].strip()

            SRL_es[(query_frame_type, query_frame_str_es)] = list()
            for arg in SRL[frame]:
                arg_type = arg[0]
                arg_str = [arg[1]]
                arg_str_embedding = model.encode(arg_str)
                closest_n = 1
                distances = scipy.spatial.distance.cdist([arg_str_embedding[0]], sentence_embeddings, "cosine")[0]
                results = zip(range(len(distances)), distances)
                results = sorted(results, key=lambda x: x[1])
                for idx, distance in results[0:closest_n]:
                    arg_str_es = subfrases[idx].strip()
                SRL_es[(query_frame_type, query_frame_str_es)].append((arg_type, arg_str_es))

        with io.open(DOWNLOAD_FOLDER + '/es_annotated_' + name, 'w', encoding='utf8') as f:
            SRL_es_pretty = pprint.pformat(SRL_es, indent=4, width=200)
            f.write(SRL_es_pretty)
            f.close()

    make_archive(DOWNLOAD_FOLDER, DOWNLOAD_FOLDER + '/' + 'es_annotated.zip')
    return redirect(url_for('download_file_frame_ann_en', filename='es_annotated.zip'))

    # return render_template('upload_frame_ann_en.html')


@app.route("/downloadfile/<filename>", methods = ['GET'])
def download_file(filename):
    return render_template('download.html', value=filename)

@app.route("/downloadfile-frame-ann-en/<filename>", methods = ['GET'])
def download_file_frame_ann_en(filename):
    return render_template('download_frame_ann_en.html', value=filename)

@app.route('/return-files/<filename>')
def return_files_tut(filename):
    file_path = DOWNLOAD_FOLDER + '/en_' + filename
    return send_file(file_path, as_attachment=True, attachment_filename='en_' + filename, cache_timeout=0)

@app.route('/return-files-frame-ann-en/<filename>', methods = ['GET'])
def return_files_tut_2(filename):
    file_path = DOWNLOAD_FOLDER + '/es_annotated.zip'
    return send_file(file_path, as_attachment=True, cache_timeout=0)


if __name__ == "__main__":
    secret = secrets.token_urlsafe(32)
    app.secret_key = secret
    app.run(host='0.0.0.0', port="5001")
