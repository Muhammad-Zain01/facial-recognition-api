from flask import Flask,render_template, request
from flask_cors import CORS
import base64
import json
import os
from main import Registration
from main import Attendence
import random
import contants as ct
import numpy as np

modal = 'TunnedModel.pickle'
path, attendence_path, model_name = ct.TEMP_PATH, ct.TEMP_ATTENDENCE_PATH, ct.MODEL_DIR+'/'+modal

RegistrationObject = Registration(path, model_name)
AttendenceObject = Attendence(attendence_path, model_name)
root_path = os.path.join(__file__, os.path.dirname(__file__))+"/temp"

if(not os.path.exists(f'{root_path}')):
    os.mkdir(f'{root_path}')
    
# print(AttendenceObject.classNames)

# RegistrationObject.Register()
# AttendenceObject.FaceCam()
app = Flask(__name__)
CORS(app, origins='*', supports_credentials=True, allow_headers=['Content-Type', 'Authorization'])

# Helper Functions
def get_unique_values_with_counts(arr):
    unique_values, counts = np.unique(arr, return_counts=True)
    unique_counts = dict(zip(unique_values, counts))
    return unique_counts

def get_user_index(arr, target):
    index_arr = []
    for i in range(len(arr)):
        value = arr[i][1]
        if(str(value) == str(target)):
            index_arr.append(i)
    return index_arr

def remove_index(arr, remove_arr):
    data = [value for index, value in enumerate(arr) if index not in remove_arr]
    return data

# @app.route('/')
# def home():
#     return render_template('Register.html')

@app.route('/trained_users', methods=['GET'])
def trained_users():
    AttendenceObject = Attendence(attendence_path, model_name);
    trainedUsers = (AttendenceObject.classNames)
    array_2d = np.array([[1, 2, 3], [4, 2, 1], [3, 4, 2]])
    unique_counts = get_unique_values_with_counts(trainedUsers)
    trainedUsers = [x for i, x in enumerate(trainedUsers) if x not in trainedUsers[:i]]
    return render_template('trained_users.html', users=[trainedUsers, unique_counts])

@app.route('/remove_user', methods=['POST'])
def remove_user():
    data = request.get_json()
    id = data['id']
    AttendenceObject = Attendence(attendence_path, model_name)
    Labels = AttendenceObject.classNames
    EncodedData = AttendenceObject.encodeListKnown
    index_to_remove = get_user_index(Labels, id)

    newLabels = remove_index(Labels, index_to_remove)
    newEncodings = remove_index(EncodedData, index_to_remove)
    RegistrationObject.modelUpdate(newLabels, newEncodings)
    return json.dumps({'Response' : "SUCCESS"})

@app.route('/upload', methods=['POST'])
def Upload():
    data = json.loads(request.form.get('data'))
    name = data['name']
    id = data['id']
    base64_string = data['img']
    img = base64_string.split(',', 1)[-1]
    image = base64.b64decode(img)
    
    if(not os.path.exists(f'{root_path}/{id}')):
        os.mkdir(f'{root_path}/{id}')
    
    with open(f'{root_path}/{id}/{name}_{id}.jpg', 'wb') as file:
        file.write(image)
    
    return json.dumps({'Response' : "SUCCESS"})
    
@app.route('/register', methods=['POST'])
def Register():
    AttendenceObject = Attendence(attendence_path, model_name)

    data = json.loads(request.form.get('data'))
    name = data['name']
    id = data['id']
    base64_string = data['img']
    img = base64_string.split(',', 1)[-1]
    image = base64.b64decode(img)
    
    
    if(not os.path.exists(f'{root_path}/{id}')):
        os.mkdir(f'{root_path}/{id}')
    
    with open(f'{root_path}/{id}/{name}_{id}.jpg', 'wb') as file:
        file.write(image)
    
    
    RegistrationObject.Register(id)
    return "HERE"
    # aReturn = {'ResponseMessage': "SUCCESS"}
    # return json.dumps(aReturn)

@app.route('/image', methods=['POST'])
def process_form_data():
    # data = request.get_json()
    # image_file = data.get('images')
    # name = data.get('name')
    # user_id = data.get('userId')
        
    # for i, image_data in enumerate(image_file):
    #     image_position = image_data[0]
    #     image_bytes = base64.b64decode(image_data[1])
    #     with open(f'{path}/{name}_{image_position}_{user_id}.jpg', 'wb') as file:
    #         file.write(image_bytes)
  

    RegistrationObject.Register()
    
    aResponse = {"ResponseCode" : "200", "ResponseMessage" : "Success"}
    return json.dumps(aResponse)
    


@app.route('/check-user', methods=['POST'])
def checkUser():    
    image = request.form.get('img')
    image_name = 'check_user.jpeg'
    with open(f'{attendence_path}/{image_name}', 'wb') as file:
        base64_string = image.split(',', 1)[-1]
        image_bytes = base64.b64decode(base64_string)
        file.write(image_bytes)
        
    AttendenceObject = Attendence(attendence_path, model_name);
    bResult = AttendenceObject.compareFaces(image_name)
    User = ""
    ResponseCode = 400
    ResponseMessage = ""
    Id = ''
    User = ''
    data = {}
    if bResult == 1:
        if len(AttendenceObject.checkUser()) > 0:
            ResponseCode = 200
            User = AttendenceObject.checkUser()[0]
            Id = AttendenceObject.checkUser()[1]
            data = {'EmployeeName' : User, "EmployeeId" : Id}
        else:
            ResponseCode = 400
            ResponseMessage = 'User Not Found'
    elif bResult == 2:
        print("user NOT DETECTED")
        ResponseCode = 400
        ResponseMessage = 'User Not Found'
    elif bResult == 3:
        print("FACE NOT DETECTED")
        ResponseCode = 400
        ResponseMessage = 'Face Not Detected'
        
    aResponse = {"ResponseCode" : ResponseCode, "ResponseMessage": ResponseMessage, "data" : data}
    return json.dumps(aResponse)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
