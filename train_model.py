import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# データのパス設定
data_dir = 'data'  # data/ フォルダのパス
categories = ['pre_harvest', 'harvest', 'post_harvest']  # 収穫前、収穫時期、収穫後のカテゴリ名

# 画像データとラベルをリストに格納
def load_images():
    images = []
    labels = []
    for category in categories:
        category_path = os.path.join(data_dir, category)
        label = categories.index(category)  # ラベルはカテゴリのインデックス
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = image.load_img(img_path, target_size=(150, 150))  # 画像をリサイズ
            img_array = image.img_to_array(img)  # 画像をNumPy配列に変換
            images.append(img_array)
            labels.append(label)
    return np.array(images), np.array(labels)

# 画像とラベルの読み込み
images, labels = load_images()

# 画像の正規化
images = images / 255.0

# データを訓練用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# モデルの構築
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3つのクラスを分類
])

# モデルのコンパイル
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# モデルの学習
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 学習結果の保存
model.save('model/harvest_model.h5')

# 学習の進行状況をプロット
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
