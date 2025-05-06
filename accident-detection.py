import cv2
from detection import AccidentDetectionModel
import numpy as np
import os

model = AccidentDetectionModel("model.json", 'model_weights.h5')
font = cv2.FONT_HERSHEY_SIMPLEX

def startapplication():
    video = cv2.VideoCapture('cars.mp4')
    while True:
        ret, frame = video.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        pred, prob = model.predict_accident(roi[np.newaxis, :, :])
        if(pred == "Accident"):
            prob = (round(prob[0][0]*100, 2))
            
            # to beep when alert:
            # if(prob > 90):
            #     os.system("say beep")

            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, pred+" "+str(prob), (20, 30), font, 1, (255, 255, 0), 2)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            return
        cv2.imshow('Video', frame)  

if __name__ == '__main__':
    startapplication()

from keras.models import model_from_json
import numpy as np

class AccidentDetectionModel(object):

    class_nums = ['Accident', "No Accident"]

    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()

    def predict_accident(self, img):
        self.preds = self.loaded_model.predict(img)
        return AccidentDetectionModel.class_nums[np.argmax(self.preds)], self.preds

from camera import startapplication

startapplication()

from twilio.rest import Client
from geopy.geocoders import Nominatim
import requests

AUTH_TOKEN = "41d15cabfb62d500928137b5b8c12338"
TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155238886"  

POLICE_WHATSAPP = "whatsapp:+917305925789"
HOSPITAL_WHATSAPP = "whatsapp:+14155238886"  

def get_location():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        location = data["loc"].split(",")  
        latitude, longitude = location[0], location[1]

        geolocator = Nominatim(user_agent="geoapi")
        address = geolocator.reverse(f"{latitude}, {longitude}").address

        maps_link = f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"
        return address, maps_link
    except Exception as e:
        return "Location Unavailable", ""

def send_whatsapp(to, message):
    client = Client(ACCOUNT_SID, AUTH_TOKEN)
    try:
        msg = client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body=message,
            to=to  
        )
        print(f"WhatsApp message sent to {to}: {msg.sid}")
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")

def alert_authorities(event):
    if event == "accident_detected":
        address, maps_link = get_location()
        message = f"🚨 Emergency Alert: Accident detected!\n📍 Location: {address}\n🔗 Map: {maps_link}"

        send_whatsapp(POLICE_WHATSAPP, message)
        send_whatsapp(HOSPITAL_WHATSAPP, message)

alert_authorities("accident_detected")
