import io
import os
from googletrans import Translator
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, send_file, render_template, url_for

UPLOAD_FOLDER = '/home/elena/PycharmProjects/WordVectors/venv/main/uploads'
DOWNLOAD_FOLDER = '/home/elena/PycharmProjects/WordVectors/venv/main/downloads'
file_original_name = ''

ALLOWED_EXTENSIONS = {'txt'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.stream.seek(0)

            with io.open(UPLOAD_FOLDER + "/" + filename, 'r', encoding='utf8') as f:
                text = f.readlines()
            translator = Translator()
            with io.open(DOWNLOAD_FOLDER + '/en_' + filename, 'w', encoding='utf8') as f:
                for sentence in text:
                    translation = translator.translate(text=sentence, src='es', dest='en')
                    f.write(translation.text + '\n')

            return redirect(url_for('download_file', filename=filename))

    return render_template('main.html')


@app.route("/downloadfile/<filename>", methods = ['GET'])
def download_file(filename):
    return render_template('download.html', value=filename)

@app.route('/return-files/<filename>')
def return_files_tut(filename):
    file_path = DOWNLOAD_FOLDER + '/en_' + filename
    return send_file(file_path, as_attachment=True, attachment_filename='en_' + filename, cache_timeout=0)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port="5001")
