import io
import os
import json
from googletrans import Translator
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, send_file, render_template, url_for

UPLOAD_FOLDER = '/home/elena/PycharmProjects/WordVectors/venv/main/uploads'
DOWNLOAD_FOLDER = '/home/elena/PycharmProjects/WordVectors/venv/main/downloads'
file_original_name = ''

ALLOWED_EXTENSIONS_txt = {'txt'}
ALLOWED_EXTENSIONS_conll = {'conll'}
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
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_txt):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'es_sentences.txt'))
            file.stream.seek(0)

            with io.open(UPLOAD_FOLDER + "/" + 'es_sentences.txt', 'r', encoding='utf8') as f:
                text = f.readlines()
            translator = Translator()
            with io.open(DOWNLOAD_FOLDER + '/en_' + filename, 'w', encoding='utf8') as f:
                for sentence in text:
                    translation = translator.translate(text=sentence, src='es', dest='en')
                    f.write(translation.text + '\n')

            return redirect(url_for('download_file', filename=filename))

    return render_template('main.html')


@app.route('/upload-frame-ann-en', methods=['GET', 'POST'])
def upload_frame_ann_en():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_conll):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.stream.seek(0)

            with io.open(UPLOAD_FOLDER + "/" + filename, 'r', encoding='utf8') as f:
                texto = f.read()
                f.seek(0)
                texto_iter_lineas = f.readlines()
                f.seek(0)

                SRL = dict()

                anotaciones = texto.split('\n\n')
                for anotacion in anotaciones:
                    lineas = anotacion.split('\n')

                    # Getting the frame identificator of the anotacion, which may be after the first labelled role
                    for linea in lineas:
                        if len(linea.split('\t')) > 3:
                            frame_id = linea.split('\t')[-3]
                            frame_type = linea.split('\t')[-2]
                            if frame_id != '_':
                                anotacion_frame = frame_type
                                SRL[anotacion_frame] = list()
                                break

                    n_linea = 0
                    for linea in lineas:
                        if len(linea.split('\t')) > 1:
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
                                        SRL[anotacion_frame].append((SRL_tag, ann_tmp))
                                        break


            with io.open(DOWNLOAD_FOLDER + '/annotated_es_' + filename, 'w', encoding='utf8') as f:
                for sentence in texto_iter_lineas:
                    f.write(str(SRL) + sentence + '\n')
                f.close()


            return redirect(url_for('download_file_frame_ann_en', filename = filename))

    return render_template('upload_frame_ann_en.html')



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

@app.route('/return-files-frame-ann-en/<filename>')
def return_files_tut_2(filename):
    file_path = DOWNLOAD_FOLDER + '/annotated_es_' + filename
    return send_file(file_path, as_attachment=True, attachment_filename='annotated_es_' + filename, cache_timeout=0)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port="5000")
