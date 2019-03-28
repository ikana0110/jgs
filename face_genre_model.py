from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------
# モデルを構築する
# ----------------------------------------

# 分類対象のカテゴリー
root_dir = "./image/"
categories = ["gal", "natural", "office", "street"]
nb_classes = len(categories)  # クラスの総数(今回は4クラス)
image_size = 100  # 画像サイズ
fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))  # プロット用変数

# データをロード
def main():
    # データのロード(xが説明変数, yが目的変数)
    X_train, X_test, y_train, y_test = np.load("./image/face3.npy")
    # データを正規化する(Kerasで読込ができるようにする)
    # RGBを表す各ピクセルデータを256で割ってデータを0から1の範囲にする
    X_train = X_train.astype("float") / 256  # 訓練用データ
    X_test = X_test.astype("float") / 256  # テストデータ
    y_train = np_utils.to_categorical(y_train, nb_classes)  # 訓練用ラベル
    y_test = np_utils.to_categorical(y_test, nb_classes)  # テストラベル
    # モデルを訓練し評価する
    model = model_train(X_train, y_train, X_test, y_test)
    model_eval(model, X_test, y_test)
    # モデルを描画する
    # plot_model(model)

    # loss/accをグラフ化する
    plot_history_loss()
    plot_history_acc()
    fig.savefig('./face-model_tutorial4.png')
    plt.close()

# モデルの構築(CNN) この辺はまだ勉強中・・・
def build_model(in_shape):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, kernel_initializer='uniform'))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=in_shape))  # 畳み込み層
    # model.add(Activation('relu'))  # 活性化関数(ReLU)
    # model.add(MaxPooling2D(pool_size=(2, 2)))  # プーリング層
    # model.add(Dropout(0.25))  # 入力にドロップアウトを適用する(過学習の防止)
    # model.add(Conv2D(64, 3, 3, border_mode='same'))  # 畳み込み層
    # model.add(Activation('relu'))  # 活性化関数(ReLU)
    # model.add(Conv2D(64, 3, 3))  # 畳み込み層
    # model.add(MaxPooling2D(pool_size=(2, 2)))  # プーリング層
    # model.add(Dropout(0.25))  # 入力にドロップアウトを適用する(過学習の防止)
    # model.add(Flatten())  # 入力を平滑化する
    # model.add(Dense(512))  # 全結合ニューラルネットワークレイヤー
    # model.add(Activation('relu'))  # 活性化関数(ReLU)
    # model.add(Dropout(0.5))  # 入力にドロップアウトを適用する(過学習の防止)
    # model.add(Dense(nb_classes))  # 全結合ニューラルネットワークレイヤー
    # model.add(Activation('softmax'))  # 活性化関数(softmax)
    # model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # model.add(Conv2D(filters=32,  # フィルター数
    #                  kernel_size=3,  # フィルターサイズ
    #                  strides=3,  # ストライド(単一の整数の場合は幅と高さが同様のストライド)
    #                  border_mode='same',  # 入出力サイズが同じ
    #                  activation='relu',  # 活性化関数(ReLU)
    #                  input_shape=in_shape  # 入力サイズ
    #                  ))  # 畳み込み層
    #
    # model.add(MaxPooling2D(pool_size=(2, 2)))  # プーリング層
    #
    # model.add(Dropout(0.25))  # 入力にドロップアウトを適用する(過学習の防止)
    #
    # model.add(Conv2D(filters=32,   # フィルター数
    #                  kernel_size=3,   # フィルターサイズ
    #                  strides=3,  # ストライド(単一の整数の場合は幅と高さが同様のストライド)
    #                  border_mode='same',  # 入出力サイズが同じ
    #                  activation='relu',  # 活性化関数(ReLU)
    #                  ))  # 畳み込み層
    #
    # model.add(MaxPooling2D(pool_size=(2, 2)))  # プーリング層
    #
    # model.add(Dropout(0.25))  # 入力にドロップアウトを適用する(過学習の防止)
    #
    # model.add(Conv2D(filters=32,   # フィルター数
    #                  kernel_size=3,   # フィルターサイズ
    #                  strides=3,  # ストライド(単一の整数の場合は幅と高さが同様のストライド)
    #                  border_mode='same',  # 入出力サイズが同じ
    #                  activation='relu',  # 活性化関数(ReLU)
    #                  ))  # 畳み込み層
    #
    # # model.add(MaxPooling2D(pool_size=(2, 2)))  # プーリング層
    #
    # model.add(Dropout(0.25))  # 入力にドロップアウトを適用する(過学習の防止)
    #
    # model.add(Flatten())  # 入力を平滑化する(一次元にする)
    # model.add(Dense(512))  # 全結合ニューラルネットワークレイヤー(通常の結合層)
    # model.add(Activation('relu'))  # 活性化関数(ReLU)
    #
    # model.add(Dropout(0.5))  # 入力にドロップアウトを適用する(過学習の防止)
    #
    # model.add(Dense(nb_classes))  # 全結合ニューラルネットワークレイヤー
    #
    # model.add(Activation('softmax'))  # 活性化関数(softmax)
    # # model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

# モデルを訓練する
def model_train(X, y, X_test, y_test):
    model = build_model(X.shape[1:])
    # batch_size=1回の学習単位？ batch_size=学習回数？ ここはどう設定すればいいか分からない
    global fit
    fit = model.fit(X, y, batch_size=32, nb_epoch=10, validation_data=(X_test, y_test))
    # モデルを保存する
    hdf5_file = "./image/face-model4.hdf5"
    model.save_weights(hdf5_file)  # モデルの重み？をHDF5形式のファイルに保存

    return model

# モデルを評価する
def model_eval(model, X, y):
    # モデルの損失値と評価値を計算 (x:入力データ, y:ラベル)
    score = model.evaluate(X, y)
    print('loss=', score[0])  # 損失率
    print('accuracy=', score[1])  # 正解率

# lossをグラフ化する
def plot_history_loss():
    axL.plot(fit.history['loss'], label="train")
    axL.plot(fit.history['val_loss'], label="val")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

# accをグラフ化する
def plot_history_acc():
    axR.plot(fit.history['acc'], label="train")
    axR.plot(fit.history['val_acc'], label="val")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='upper right')

# モデルの可視化
def plot_model(model):
    plot_model(model, to_file='face_genre_model.png')

if __name__ == "__main__":
    main()
