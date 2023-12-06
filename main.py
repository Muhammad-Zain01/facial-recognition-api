import cv2
import numpy as np
import face_recognition
import os
import pickle
import shutil
        
        
class Registration:
    def __init__(self, model_name):
        self.modelName = model_name
        self.root = os.path.join(__file__,  os.path.dirname(__file__))
        self.model = f'{self.root}/{self.modelName}'
        self.images = []
        self.image_labels = []
        self.classNames = []
        self.bModel = False
        self.encodedData = False

    def checkModel(self):
        if os.path.exists(self.model):
            self.bModel = True
            with open(self.model, 'rb') as file:
                self.encodedData = pickle.load(file)

    def getImages(self, path):
        myList = os.listdir(f'{self.root}/temp/{path}')
        for cls in myList:
            curImg = cv2.imread(f'{self.root}/temp/{path}/{cls}')
            self.images.append(curImg)
            tarr = os.path.splitext(cls)[0].split('_')
            self.image_labels.append([tarr[0],tarr[1]])
        return True

    def encodeit(self):
        encodelist = []
        i = 0
        for image in self.images:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(img)
            if len(face_locations) > 0:
                encode = face_recognition.face_encodings(img, face_locations)
                encodelist.append(encode[0])
                self.classNames.append(self.image_labels[i])
            i += 1
        return encodelist

    def Reset(self, path):
        self.removeDirs(path)
        self.images = []
        self.classNames = []
        self.encodedData = []
        return True

    def removeDirs(self, path):
        f'{self.root}/temp/{path}'
        file_path = os.path.join(self.root, 'temp', path)
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)
        return True

    def modelUpdate(self, classNames, encodeListKnown):
        with open(self.modelName, 'wb') as file:
            pickle.dump([classNames, encodeListKnown], file)

    def Register(self, path):
        self.checkModel()
        self.getImages(path)
        
        encodeListKnown = self.encodeit()
        if self.bModel:
            oldClassNames = self.encodedData[0]
            oldFaceData = self.encodedData[1]

            for names in self.classNames:
                oldClassNames.append(names)

            for facedata in encodeListKnown:
                oldFaceData.append(facedata)

            encodeListKnown = oldFaceData
            self.classNames = oldClassNames
        
        with open(self.model, 'wb') as file:
            pickle.dump([self.classNames, encodeListKnown], file)

        self.Reset(path)
        return True

class Detection:
    def __init__(self, temp_path, modelName):
        self.root = os.path.join(__file__,  os.path.dirname(__file__))
        self.temp = temp_path
        self.modelName = modelName
        self.user = []
        self.error = ''
        self.classNames = []
        self.encodeListKnown = []

        if os.path.exists(f'{self.root}/{self.modelName}'):
            with open(f'{self.root}/{self.modelName}', 'rb') as file:
                unpickler = pickle.Unpickler(file)
                encoded = unpickler.load()
            self.error = 'Model Not Found'
            self.classNames = encoded[0]
            self.encodeListKnown = encoded[1]
        
    def DetectFace(self, img):
        img = cv2.imread(img)
        test_image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(test_image_rgb)
        face_encodings = face_recognition.face_encodings(test_image_rgb, face_locations)
        faceDetected = False
        for face_encoding, face_location in zip(face_encodings, face_locations):
            faceDetected = True
            distances = face_recognition.face_distance(self.encodeListKnown, face_encoding)
            min_distance_index = np.argmin(distances)
            min_distance = distances[min_distance_index]
            threshold = 0.37 
            if min_distance <= threshold:
                label = self.classNames[min_distance_index]
                return label[0]
        if faceDetected:
            return True
        else:
            return False

    def FaceCam(self):
        facecam = cv2.VideoCapture(0)
        while True:
            bAttendence = False
            ret,frame = facecam.read()
            frameS = cv2.resize(frame,(0,0),None,0.25,0.25)
            frameS = cv2.cvtColor(frameS, cv2.COLOR_BGR2RGB)
            facesCurFrame = face_recognition.face_locations(frameS)
            encodeofCurrFrame = face_recognition.face_encodings(frameS,facesCurFrame)

            for encodeFace,faceLoc in zip(encodeofCurrFrame,facesCurFrame):
                distances = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                min_distance_index = np.argmin(distances)
                min_distance = distances[min_distance_index]
                matches = face_recognition.compare_faces(self.encodeListKnown,encodeFace)
                faceDis = face_recognition.face_distance(self.encodeListKnown,encodeFace)
                matchIndex = np.argmin(faceDis)
                distances = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                min_distance_index = np.argmin(distances)
                min_distance = distances[min_distance_index]
                threshold = 0.43
                if min_distance <= threshold:
                    if matches[matchIndex]:
                        name = self.classNames[matchIndex][0] + " - " + self.classNames[matchIndex][1].upper()
                        y1,x2,y2,x1 = faceLoc
                        y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
                        cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                        cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                        break
                    else:
                        y1,x2,y2,x1 = faceLoc
                        y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
                        cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                        cv2.putText(frame,'Unknown',(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                else:
                    y1,x2,y2,x1 = faceLoc
                    y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
                    cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                    cv2.putText(frame,'Unknown',(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            cv2.imshow("Facecam",frame)
            key = cv2.waitKey(30)
            if bAttendence:
                break
        return cv2.destroyAllWindows

    def compareFaces(self, img):
        image = self.temp+'/'+img
        Detection = self.DetectFace(image)
        return Detection
        if Detection == 'FACE_DETECTED':
            return 2
        elif Detection == 'FACE_NOT_DETECTED':
            return 3
        else:
            self.user = Detection
            return 1

    def checkUser(self):
        return self.user