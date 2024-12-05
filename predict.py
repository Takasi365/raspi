import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# モデルのロード
model = tf.keras.models.load_model('models/harvest_model.h5')

# 画像の読み込みと前処理
img_path = 'test_image.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 予測
predictions = model.predict(img_array)
class_names = ['before_harvest', 'harvest', 'over_harvest']
predicted_class = class_names[np.argmax(predictions)]

print(f'予測結果: {predicted_class}')
