import autoencoder_model as autoencoder_model
import sys, os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ----------------------------------------
# AEモデルの検証用
# ----------------------------------------
image_size = 100

# 顔認識用特徴量ファイルを読み込む --- （カスケードファイルのパスを指定）
cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')

# 入力画像をnumpyに変換
X = []
files = []

if __name__ == '__main__':
    # ここは顔判断したい画像を指定
    image_pic = cv2.imread("./hirose_suzu.jpg")

# # 顔認識の実行
face = cascade.detectMultiScale(image_pic)

if len(face) == 0:
    print("顔検出できませんでした")
    exit()

# 顔部分を切り取る
for x, y, w, h in face:
    face_cut = image_pic[y:y + h, x:x + w]

cv2.imwrite('face_cut.jpg', face_cut)

# 画像の読み込み
img = Image.open("./face_cut.jpg")
img = img.convert("RGB")
img = img.resize((image_size, image_size))

# 画像を表示
# plt.imshow(img)
# plt.show()

in_data = np.asarray(img)
X.append(in_data)

X = np.array(X)

# x_train, x_test, _, _ = np.load("./image/face3.npy")
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255

# モデル構築
autoencoder = autoencoder_model.build_model(X.shape[1:])
# autoencoder = autoencoder_model.build_model(x_train.shape[1:])
autoencoder.load_weights('autoencoder.h5')

# データを予測
decoded_imgs = autoencoder.predict(X)

# 入力画像と再現した画像を表示する
n = 1  # 20個の画像を表示する

plt.figure(figsize=(20, 4))
for i in range(n):
    # 特徴量を抽出する
    # print('特徴量', y[i].shape)
    # オリジナルのテスト画像を表示
    ax = plt.subplot(2, n, i+1)
    plt.imshow(X[i])
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 変換された画像を表示
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i])
    # plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
