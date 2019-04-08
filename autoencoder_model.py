# ----------------------------------------------------
# オートエンコーダーのモデルを構築する
# ----------------------------------------------------
from keras import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np

# データ準備
# (x_train, _), (x_test, _) = mnist.load_data()
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# x_train, x_test, _, _ = np.load("./image/face3.npy")
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255
# # x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# # x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
#
# # モデル構築
# # input_img = Input(shape=(784,))
# input_img = Input(shape=x_train.shape[1:])
#
# x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(input_img)
# x = MaxPooling2D((2, 2), border_mode='same')(x)
# x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
# x = MaxPooling2D((2, 2), border_mode='same')(x)
# x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
# encoded = MaxPooling2D((2, 2), border_mode='same', name='encoder_layer')(x)
#
# x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(encoded)
# x = UpSampling2D((2, 2))(x)
# x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
# x = UpSampling2D((2, 2))(x)
# x = Convolution2D(64, 3, 3, activation='relu')(x)
# x = UpSampling2D((2, 2))(x)
# decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)
#
# autoencoder: Model = Model(input_img, decoded)
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#
# # モデルの訓練
# # autoencoder.fit(x_train, x_train, nb_epoch=10, batch_size=32, shuffle=True, validation_data=(x_test, x_test))
#
# # モデルの保存
# # autoencoder.save_weights('autoencoder.h5')
#
# # モデルの読み込み
# autoencoder.load_weights('autoencoder.h5')
#
# # 中間層の特徴量を抽出する
# checkpoint_model = Model(autoencoder.input, autoencoder.get_layer(name='encoder_layer').output)
# y = checkpoint_model.predict(x_test)
# # print('特徴量', y.shape)
#
# decoded_imgs = autoencoder.predict(x_test)
#
# # 入力画像と再現した画像を表示する
# n = 20  # 20個の画像を表示する
#
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # 特徴量を抽出する
#     print('特徴量', y[i].shape)
#     # オリジナルのテスト画像を表示
#     ax = plt.subplot(2, n, i+1)
#     plt.imshow(x_test[i])
#     # plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # 変換された画像を表示
#     ax = plt.subplot(2, n, i+1+n)
#     plt.imshow(decoded_imgs[i])
#     # plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

# メイン処理
def main():
    # データ準備
    # (x_train, _), (x_test, _) = mnist.load_data()
    # x_train = x_train.astype('float32') / 255.
    # x_test = x_test.astype('float32') / 255.
    # x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    # x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    x_train, x_test, _, _ = np.load("./image/face3.npy")
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # モデルを訓練
    # autoencoder = train_model(x_train, x_test)

    # モデルを読み込む
    autoencoder = load_model(x_test)

    # 中間層の特徴量を抽出する
    checkpoint_model = Model(autoencoder.input, autoencoder.get_layer(name='encoder_layer').output)
    y = checkpoint_model.predict(x_test)
    # print('特徴量', y.shape)

    decoded_imgs = autoencoder.predict(x_test)

    n = 10  # 10個の画像を表示する

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # 特徴量を抽出する
        print('特徴量' + str(i + 1), y[i])
        # オリジナルのテスト画像を表示
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i])
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # 変換された画像を表示
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        # plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

# モデルの構築
def build_model(in_shape):
    input_img = Input(shape=in_shape)

    x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(input_img)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = MaxPooling2D((2, 2), border_mode='same')(x)
    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
    encoded = MaxPooling2D((2, 2), border_mode='same', name='encoder_layer')(x)

    x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Convolution2D(64, 3, 3, activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)

    autoencoder: Model = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder

# モデルを訓練する
def train_model(x_train, x_test):
    # モデルの構築
    autoencoder = build_model(x_train.shape[1:])
    # モデルの訓練
    autoencoder.fit(x_train, x_train, nb_epoch=10, batch_size=32, shuffle=True, validation_data=(x_test, x_test))
    # モデルの保存
    autoencoder.save_weights('autoencoder.h5')
    return autoencoder

# モデルを読み込む
def load_model(x_train):
    # モデルの構築
    autoencoder = build_model(x_train.shape[1:])
    # モデル読み込み
    autoencoder.load_weights('autoencoder.h5')
    return autoencoder

if __name__ == "__main__":
    main()










