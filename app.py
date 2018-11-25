import os
from flask import Flask, render_template, request, Response, json

app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # file = request.files['image']
    # f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    # file.save(f)

    part = request.args.get('part')
    # Line #1
    minX = request.args.get('minX')
    minY = request.args.get('minY')
    maxX = request.args.get('maxX')
    maxY = request.args.get('maxY')
    # Line #2
    minX2 = request.args.get('minX2')
    minY2 = request.args.get('minY2')
    maxX2 = request.args.get('maxX2')
    maxY2 = request.args.get('maxY2')

    bytesData = request.get_data()
    path = os.path.join(app.config['UPLOAD_FOLDER'], 'image.png')
    f = open(path, 'wb')
    f.write(bytesData)
    f.close()

    # TODO: Process image.
    size = 150

    data = {
        'size': size
    }
    res = Response(json.dumps(data), status=200, mimetype='application/json')
    return res
