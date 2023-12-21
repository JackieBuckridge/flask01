import os
from flask import Flask, render_template, request,send_from_directory
import time

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return "No video part"

    file = request.files['video']

    if file.filename == '':
        return "No selected video file"

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], f"{file.filename}_{time.time()}")
        file.save(filename)
        print("File path:",file)
        return 'Video uploaded successfully'

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
