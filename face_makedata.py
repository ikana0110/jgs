from PIL import Image
import os, glob
import numpy as np
import random, math

# ----------------------------------------
# 学習データを作成する
# ----------------------------------------

# 分類対象のカテゴリを選ぶ
root_dir = "./image/"
categories = ["gal", "natural", "office", "street"]
nb_classes = len(categories)
image_size = 100

# 画像データを読み込んで、numpy形式のデータに変換する
X = []  # 画像データ
Y = []  # ラベルデータ
def add_sample(cat, fname, is_train):
    img = Image.open(fname)  # 画像を開く
    img = img.convert("RGB")  # カラーモードの変更
    img = img.resize((image_size, image_size))  # 画像サイズの変更
    data = np.asarray(img)  # 配列を作成
    X.append(data)  # 画像データの配列を追加
    Y.append(cat)  # ラベルを追加

    if not is_train: return  # 訓練データではなかったら終了

    # 角度を変えた訓練データを追加
    # 少しずつ回転する
    for ang in range(-20, 20, 5):
        img2 = img.rotate(ang)
        data = np.asarray(img2)
        X.append(data)
        Y.append(cat)
        # 反転する
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
        data = np.asarray(img2)
        X.append(data)
        Y.append(cat)

# データを作成し、numpy形式のデータを返す
def make_sample(files, is_train):
    global X, Y
    X = []; Y = []
    for cat, fname in files:
        add_sample(cat, fname, is_train)
    return np.array(X), np.array(Y)

# ディレクトリごとに分けられたファイルを収集する
allfiles = []
for idx, cat in enumerate(categories):  # cat:カテゴリ名
    image_dir = root_dir + "/" + cat
    files = glob.glob(image_dir + "/*.jpg")
    for f in files:
        allfiles.append((idx, f))

# シャッフルして学習データとテストデータに分ける
random.shuffle(allfiles)  # ファイルの一覧をシャッフル
th = math.floor(len(allfiles) * 0.6)  # 参考書のパクリ。なんで？？
train = allfiles[0:th]  # 訓練データ
test = allfiles[th:]  # テストデータ
X_train, y_train = make_sample(train, True)  # 訓練データは水増し
X_test, y_test = make_sample(test, False)  # テストデータ
xy = (X_train, X_test, y_train, y_test)
np.save("./image/face3.npy", xy)  # numpy形式のデータを保存
print("ok,", len(y_train))