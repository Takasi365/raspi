import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# 学習済みモデルのロード
model = tf.keras.models.load_model('models/harvest_model.h5')

# 予測する画像ファイルのパス
img_path = 'data/test_images/test_image.jpg'

# 画像の読み込みと前処理
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # バッチの次元を追加
img_array = img_array / 255.0  # 正規化

# 予測
predictions = model.predict(img_array)
class_idx = np.argmax(predictions)  # 最も確率が高いクラスを選択

# クラスラベルの設定
classes = ['before_harvest', 'harvest', 'over_harvest']
predicted_class = classes[class_idx]

print(f"Predicted harvest stage: {predicted_class}")
