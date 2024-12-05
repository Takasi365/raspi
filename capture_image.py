import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from picamera2 import Picamera2
import time

# 学習済みモデルの読み込み
model = tf.keras.models.load_model('model/harvest_model.h5')

# カメラのセットアップ
picam2 = Picamera2()
picam2.start()

# 画像キャプチャの関数
def capture_image():
    # 画像をキャプチャ
    picam2.capture_file("captured_image.jpg")

    # 画像の読み込みと前処理
    img = image.load_img('captured_image.jpg', target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # バッチサイズを追加
    img_array = img_array / 255.0  # 正規化

    # モデルによる予測
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)

    # 結果の表示
    categories = ['pre_harvest', 'harvest', 'post_harvest']
    print(f"予測された収穫状態: {categories[predicted_class]}")

# 画像をキャプチャして収穫状態を判定
capture_image()

# 終了
picam2.stop()
