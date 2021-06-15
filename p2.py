import os
import cv2
from keras.models import model_from_json
from keras.preprocessing import image
from datetime import datetime
from keras.preprocessing.image import ImageDataGenerator

import numpy as np





face_cascade = cv2.CascadeClassifier(r'C:\Users\hp\PycharmProjects\FER\src\model\haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)


emotions = ('anger', 'disgust', 'fear', 'happy', 'sadness', 'surprise', 'contempt')

from flask import *
import pymysql
from werkzeug.utils import secure_filename

app=Flask(__name__)
#con=pymysql.connect(host="localhost",user="root",password="root",port=3306,db="qr")
#cmd=con.cursor()
app.secret_key="aaaa"
@app.route('/')
def login():
    return render_template('code2.html')
@app.route('/image_upload',methods=['post'])
def img_upload():
    f=request.files['upload']
    fn=secure_filename(f.filename)
    f.save("static/images/"+fn)
    model = model_from_json(
        open(r"C:\Users\hp\PycharmProjects\FER\src\model\facial_expression_model_structure.json", "r").read())
    model.load_weights(r'C:\Users\hp\PycharmProjects\FER\src\model\facial_expression_model_weights.h5')  # load weights
    img=cv2.imread("static/images/"+fn)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # print(faces) #locations of detected faces
    emotion = None

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # draw rectangle to main image

        detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)  # transform to gray scale
        detected_face = cv2.resize(detected_face, (48, 48))  # resize to 48x48

        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)

        img_pixels /= 255  # pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

        predictions = model.predict(img_pixels)  # store probabilities of 7 expressions

        # find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
        max_index = np.argmax(predictions[0])

        emotion = emotions[max_index]
        cv2.putText(img, emotion, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        print(emotion)
    fn=datetime.now().strftime("%Y%m%d%H%M%S")+".jpg"
    cv2.imwrite("static/images/"+fn,img)
    return render_template("result.html",f=fn)
app.run(debug=True)