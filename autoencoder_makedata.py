from PIL import Image
import os, glob
import numpy as np
import random, math

# ----------------------------------------
# 学習データを作成する
# ----------------------------------------
root_dir = "./image/autoencoder"
image_size = 100
X = []
Y = []

# ファイルのリストを取得
files = os.listdir(root_dir)

for file in files:
    img = Image.open(root_dir + "/" + file)  # 画像を開く
    img = img.convert("RGB")  # カラーモードの変更
    img = img.resize((image_size, image_size))  # 画像サイズの変更
    data = np.asarray(img)  # 配列を作成
    X.append(data)
    Y.append(data)

    # # 左右反転
    # img2 = img.transpose(Image.FLIP_LEFT_RIGHT)
    # data = np.asarray(img2)
    # X.append(data)
    # Y.append(data)
    #
    # # 上下反転
    # img2 = img.transpose(Image.FLIP_TOP_BOTTOM)
    # data = np.asarray(img2)
    # X.append(data)
    # Y.append(data)
    #
    # # 90度回転
    # img2 = img.transpose(Image.ROTATE_90)
    # data = np.asarray(img2)
    # X.append(data)
    # Y.append(data)
    #
    # # 270度回転
    # img2 = img.transpose(Image.ROTATE_270)
    # data = np.asarray(img2)
    # X.append(data)
    # Y.append(data)

    # for ang in range(-20, 20, 5):
    #     img2 = img.rotate(ang)
    #     data = np.asarray(img2)
    #     X.append(data)
    #     Y.append(data)
    #     # 反転する
    #     img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
    #     data = np.asarray(img2)
    #     X.append(data)
    #     Y.append(data)

X = np.array(X)
Y = np.array(Y)
xy = (X, Y)

np.save("./image/autoencoder_data3.npy", xy)  # numpy形式のデータを保存

print("ok", len(Y))
