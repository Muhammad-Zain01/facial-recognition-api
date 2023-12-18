from flask import Flask, request
from flask_cors import CORS
import base64
import json
import os
from main import Registration
from main import Detection
import random

modelDir = 'models'
model_name = f'{modelDir}/TunnedModel.pickle'
root_path = os.path.join(__file__, os.path.dirname(__file__))
temp_path = os.path.join(__file__, os.path.dirname(__file__))+"/temp"
temp_path2 = os.path.join(__file__, os.path.dirname(__file__))+"/temp2"

if(not os.path.exists(f'{temp_path}')): os.mkdir(f'{temp_path}')
if(not os.path.exists(f'{temp_path2}')): os.mkdir(f'{temp_path2}')
if(not os.path.exists(f'{root_path}/{modelDir}')): os.mkdir(f'{root_path}/{modelDir}')

app = Flask(__name__)
CORS(app, origins='*', supports_credentials=True, allow_headers=['Content-Type', 'Authorization'])

@app.route('/upload', methods=['POST'])
def Upload():
    data = json.loads(request.form.get('data'))
    name = data['name']
    id = data['id']
    base64_string = data['img']
    img = base64_string.split(',', 1)[-1]
    image = base64.b64decode(img)
    random_digit = random.randint(1, 1000000)
    if(not os.path.exists(f'{temp_path}/{id}')):
        os.mkdir(f'{temp_path}/{id}')
    
    with open(f'{temp_path}/{id}/{name}_{id}{random_digit}.jpg', 'wb') as file:
        file.write(image)
    
    return json.dumps({'Status' : 1})
    
@app.route('/register', methods=['POST'])
def Register():
    path = request.form.get('path')
    RegistrationObject = Registration(model_name)
    RegistrationObject.Register(path)

    aReturn = {'Status': 1}
    return json.dumps(aReturn)

@app.route('/check', methods=['POST'])
def checkUser():
    image = request.form.get('img')
    num = random.randint(1, 10000)
    image_name = f'{num}.jpeg'
    
    with open(f'{temp_path2}/{image_name}', 'wb') as file:
        base64_string = image.split(',', 1)[-1]
        image_bytes = base64.b64decode(base64_string)
        file.write(image_bytes)
        
    DetectionObject = Detection(temp_path2, model_name);
    bResult = DetectionObject.compareFaces(image_name)
    
    if(bResult): aResponse = {"Status" : 1, "user" : bResult}
    else: aResponse = {"Status" : 0}
    
    sResponse = json.dumps(aResponse)
    return sResponse

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
