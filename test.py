import cv2
import numpy as np

from tensorflow.keras.models import model_from_json

class FacialExpressionModel(object):
    EMOTIONS_LIST = ["Angry",
                     "Disgust",
                     "Fear",
                     "Happy",
                     "Sad",
                     "Surprise",
                     "Neutral"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        print("Model loaded from disk")
        self.loaded_model.summary()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)

        return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]


#if __name__ == '__main__':
    #pass


facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX


def __get_data__():

    img_array = cv2.imread("dp.jpeg")  # read in the image, convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray, 1.1, 10)

    return faces, img_array, gray

def start_app(cnn):
    skip_frame = 10
    data = []
    flag = False
    ix = 0
    while True:
        ix += 1

        faces, fr, gray_fr = __get_data__()
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y + h, x:x + w]

            roi = cv2.resize(fc, (48, 48))
            pred = cnn.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('pred', fr)
    cv2.destroyAllWindows()

model = FacialExpressionModel("fer_model.json", "fer_weights.h5")
start_app(model)