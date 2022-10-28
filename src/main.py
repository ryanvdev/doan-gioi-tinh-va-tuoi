import datetime
import time
import uuid
import cv2
import argparse
from flask import Flask, request
import numpy as np
from PIL import Image
from flask_cors import CORS
import base64
import json
import os
from waitress import serve

ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp']

app = Flask(__name__, static_folder='../public', static_url_path='/public')
app.config['UPLOAD_FOLDER'] = 'images'
cors = CORS(app)
app.secret_key = 'super secret key 274554'
app.config['SESSION_TYPE'] = 'filesystem'


parser=argparse.ArgumentParser()
parser.add_argument('--image')

args=parser.parse_args()

faceProto="./data/opencv_face_detector.pbtxt"
faceModel="./data/opencv_face_detector_uint8.pb"
ageProto="./data/age_deploy.prototxt"
ageModel="./data/age_net.caffemodel"
genderProto="./data/gender_deploy.prototxt"
genderModel="./data/gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

video=cv2.VideoCapture(args.image if args.image else 0)
padding=20



def readImageFromBase64(base64Image):
   encoded_data = base64Image.split(',')[1]
   npArr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(npArr, cv2.IMREAD_COLOR)
   return img


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            # cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


def predict(frame):
    resultImg, faceBoxes=highlightFace(faceNet,frame)
    if not faceBoxes:
        print("No face detected")
    
    result = []

    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]
        # print(f'Gender: {gender}')

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        # print(f'Age: {age[1:-1]} years')

        # cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
    
        result.append({
            'bbox': {
                'x1': faceBox[0],
                'y1': faceBox[1],
                'x2': faceBox[2],
                'y2': faceBox[3]
            },
            'gender': gender,
            'age': age
        })

    return result



@app.route('/', methods=['GET'])
def index_router():
    return {"foo": "bar"}, 200, {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Methods": "*"
    }



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def makeDefaultHeader():
    return {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'content-type',
        'Access-Control-Allow-Method': 'POST, HEAD, OPTIONS'
    }

@app.route('/predict', methods=['OPTION', 'HEAD', 'POST'])
def predict_route():
    if request.method == 'POST':
        formData = request.get_json()
        imageExtension:str = formData['imageExtension']
        img = readImageFromBase64(formData['data'])
        
        h, w, _ = img.shape
        delta = w/h
        newWidth = 600
        newHeight = int(round(newWidth/delta, 0))

        imgProcessed = cv2.resize(img, dsize=(newWidth, newHeight))

        startTime = time.time()
        predictResult = predict(imgProcessed)
        endTime = time.time()
        
        imgProcessed = cv2.cvtColor(imgProcessed, cv2.COLOR_BGR2RGB)

        _id = uuid.uuid4()
        imgInfo = {
            'predictResult': predictResult,
            'imageName': formData['imageName'],
            'imageSize': {
                'width': newWidth,
                'height': newHeight
            },
            'createdAt': str(datetime.datetime.now().isoformat()),
            'predictTime': str(endTime - startTime),
            '_id': str(_id)
        }

        # Save image info
        f = open(f'./public/info/{_id}.json', 'w')
        f.write(json.dumps(imgInfo, indent=4))
        f.close()

        # Save image
        Image.fromarray(imgProcessed).save(
            f'./public/images/{_id}.webp', #
            format='webp'
        )

        return imgInfo, 201, makeDefaultHeader()


    return '', 202, makeDefaultHeader()
    



def development(port):
    app.run(host="0.0.0.0", port=port, debug=True)


def production(port):
    serve(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    FLASK_ENV = os.environ.get('FLASK_ENV') or 'development'
    PORT = os.environ.get('PORT') or 80

    if(FLASK_ENV == 'development'):
        development(PORT)

    elif(FLASK_ENV == 'production'):
        production(PORT)
