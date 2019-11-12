from waste_recognition import recognize
import cv2
import numpy as np


if __name__ == '__main__':
    recognizer = recognize.Model(17, label=[i for i in range(17)])
    x = cv2.imread('/media/palm/data/MicroAlgae/16_8_62/images/MIF eggs-kato-40x (1).jpg')
    y = recognizer.predict(x, bgr_to_rgb=True)
    print(y)
