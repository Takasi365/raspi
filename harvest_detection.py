import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from picamera2 import Picamera2
import time

# 学習済みモデルの読み込み関数
def load_or_train_model(data_dir, model_path='model/harvest_model.h5'):
    # モデルが存在すれば読み込む
    if os.path.exists(model_path):
        print("既存のモデルを読み込んでいます...")
        model = tf.keras.models.load_model(model_path)
    else:
        print("新しいモデルを学習します...")
        # 画像データとラベルを読み込む
        images, labels = load_images(data_dir)
        
        # 画像データの正規化
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
        os.makedirs('model', exist_ok=True)
        model.save(model_path)

        # 学習の進行状況をプロット
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.show()

    return model

# 画像データとラベルをリストに格納する関数
def load_images(data_dir):
    categories = ['pre_harvest', 'harvest', 'post_harvest']  # カテゴリ名
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

# 画像をキャプチャして収穫状態を判定する関数
def capture_image_and_predict(model):
    # カメラのセットアップ
    picam2 = Picamera2()
    picam2.start()

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

    # 終了
    picam2.stop()

# 主な処理
def main():
    data_dir = 'data'  # 画像データの保存場所
    model_path = 'model/harvest_model.h5'  # モデルの保存先パス

    # モデルの読み込みまたは学習
    model = load_or_train_model(data_dir, model_path)

    # 画像をキャプチャして収穫状態を判定
    capture_image_and_predict(model)

if __name__ == '__main__':
    main()
