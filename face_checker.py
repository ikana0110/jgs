import face_genre_model as face_model
from keras.models import load_model
import sys, os
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ----------------------------------------
# モデルを読み込んで、顔ジャンルを判断する
# ----------------------------------------

# # コマンドラインからファイル名を取得
# if len(sys.argv) <= 1:
#     print("face_checker.py")
#     quit()

image_size = 100
categories = ["gal", "natural", "office", "street"]

# 顔認識用特徴量ファイルを読み込む --- （カスケードファイルのパスを指定）
cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')

# 入力画像をnumpyに変換
X = []
files = []

if __name__ == '__main__':
    # ここは顔判断したい画像を指定
    image_pic = cv2.imread("./xxxx.jpg")

# グレースケールに変換する
# img_g = cv2.cvtColor(cv2.imread(cv2.imread("./gal.jpg"), cv2.COLOR_BGR2RGB))

# 顔認識の実行
face = cascade.detectMultiScale(image_pic)

if len(face) == 0:
    print("顔検出できませんでした")
    exit()

# 顔部分を切り取る
for x, y, w, h in face:
    face_cut = image_pic[y:y + h, x:x + w]

# face_list_1 = face_list[0].convert("RGB")
# face_list_2 = face_list[0].resize((image_size, image_size))

# face_cut = face_cut.resize((image_size, image_size))

cv2.imwrite('face_cut.jpg', face_cut)

# 画像の読み込み
img = Image.open("./face_cut.jpg")
img = img.convert("RGB")
img = img.resize((image_size, image_size))

plt.imshow(img, cmap='gray')
plt.show()

in_data = np.asarray(img)
X.append(in_data)

# for fname in sys.argv[1:]:
#     # img = Image.open(fname)
#     # img = img.convert("RGB")
#     # img = img.resize((image_size, image_size))
#
#     # 画像の読み込み
#     image_pic = cv2.imread("./gal.jpg")
#
#     # グレースケールに変換する
#     img_g = cv2.imread(image_pic, 0)
#
#     # 顔認識の実行
#     face = cascade.detectMultiScale(img_g)
#
#     # 顔部分を切り取る
#     for x, y, w, h in face:
#         face_cut = image_pic[y:y + h, x:x + w]
#
#     # face_list_1 = face_list[0].convert("RGB")
#     # face_list_2 = face_list[0].resize((image_size, image_size))
#
#     plt.imshow(face_cut, cmap='gray')
#     plt.show()
#
#     in_data = np.asarray(face_cut)
#     X.append(in_data)
#     files.append(fname)

X = np.array(X)

# CNNのモデル構築
model = face_model.build_model(X.shape[1:])
model.load_weights("./image/face-model4.hdf5")

# model = load_model("./image/face-model.hdf5")

# データを予測
html = ""
pre = model.predict(X)
print(pre)

for i, p in enumerate(pre):
    y = p.argmax()
    # print("+ 入力:", files[i])
    print("| 顔の雰囲気:", categories[y])
    print("| 確率:", p[y])
