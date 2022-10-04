import tensorflow
from tensorflow.keras.models import model_from_json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cvzone

import tensorflow as tf


######################
# 保存したモデル構造の読み込み
model2 = model_from_json(open("./vgg16.json", 'r').read())
# 保存した学習済みの重みを読み込み
model2.load_weights("./vgg16.hdf5")
cam_w, cam_h = 640, 480
dic = {5:'hebi',7:'hituji',11:'inoshishi',10:'inu',0:'nezumi',8:'saru',4:'tatu',2:'tora',9:'tori',6:'uma',3:'usagi',1:'ushi'}

cap = cv2.VideoCapture(0)
cap.set(3, cam_w)
cap.set(4, cam_h)
######################

def putTextRect(img, text, pos, scale=3, thickness=3, colorT=(255, 255, 255),
                colorR=(255, 0, 255), font=cv2.FONT_HERSHEY_PLAIN,
                offset=10, border=None, colorB=(0, 255, 0)):
    ox, oy = pos
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)

    x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset

    cv2.rectangle(img, (x1, y1), (x2, y2), colorR, cv2.FILLED)
    if border is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), colorB, border)
    cv2.putText(img, text, (ox, oy), font, scale, colorT, thickness)

    return img, [x1, y2, x2, y1]


while True:
    success, img = cap.read()
    imgRGB = Image.fromarray(img, 'RGB') # 画像をRGB形式に変換
    imgRGB = imgRGB.resize((45,90))      # 分類器にセットするために高さ45, 幅90に変換 
    imgRGB = np.array(imgRGB)            # numpy.arrayに変換
    imgRGB = np.expand_dims(imgRGB, axis=0) # 分類器にセットするために一次元増やす
    predict_y = model2.predict(imgRGB)      # 分類
    pred_sign = dic[predict_y.argmax()]     # 確率が最大となっているラベル
    img, bbox = putTextRect(img, pred_sign, [50, 50], 2, 2, offset=10, border=5) # ラベルを画面に表示
    cv2.imshow("Image", img)
    pressed_key = cv2.waitKey(1)
    if pressed_key == 27:
            break