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

# Helper Functions
# def get_unique_values_with_counts(arr):
#     unique_values, counts = np.unique(arr, return_counts=True)
#     unique_counts = dict(zip(unique_values, counts))
#     return unique_counts

# def get_user_index(arr, target):
#     index_arr = []
#     for i in range(len(arr)):
#         value = arr[i][1]
#         if(str(value) == str(target)):
#             index_arr.append(i)
#     return index_arr

# def remove_index(arr, remove_arr):
#     data = [value for index, value in enumerate(arr) if index not in remove_arr]
#     return data

# @app.route('/trained_users', methods=['GET'])
# def trained_users():
#     AttendenceObject = Detection(temp_path2, model_name);
#     trainedUsers = (AttendenceObject.classNames)
#     array_2d = np.array([[1, 2, 3], [4, 2, 1], [3, 4, 2]])
#     unique_counts = get_unique_values_with_counts(trainedUsers)
#     trainedUsers = [x for i, x in enumerate(trainedUsers) if x not in trainedUsers[:i]]
#     return render_template('trained_users.html', users=[trainedUsers, unique_counts])

# @app.route('/remove_user', methods=['POST'])
# def remove_user():
#     data = request.get_json()
#     id = data['id']
#     AttendenceObject = Detection(temp_path2, model_name)
#     Labels = AttendenceObject.classNames
#     EncodedData = AttendenceObject.encodeListKnown
#     index_to_remove = get_user_index(Labels, id)

#     newLabels = remove_index(Labels, index_to_remove)
#     newEncodings = remove_index(EncodedData, index_to_remove)
#     RegistrationObject.modelUpdate(newLabels, newEncodings)
#     return json.dumps({'Response' : "SUCCESS"})

@app.route('/upload', methods=['POST'])
def Upload():
    data = json.loads(request.form.get('data'))
    name = data['name']
    id = data['id']
    base64_string = data['img']
    img = base64_string.split(',', 1)[-1]
    image = base64.b64decode(img)
    random_digit = random.randint(1, 100)
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
    
    if(bResult): aResponse = {"Response" : "SUCCESS", "user" : bResult}
    else: aResponse = {"Response" : "NOT_FOUND"}
    
    sResponse = json.dumps(aResponse)
    return sResponse

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
