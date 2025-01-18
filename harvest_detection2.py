import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from picamera2 import Picamera2
import time
import serial  # Arduinoとのシリアル通信用

# シリアルポートの設定（適宜変更）
SERIAL_PORT = "/dev/ttyACM0"  # Arduinoのポート
BAUD_RATE = 9600  # ボーレート
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# モデルの保存パス
MODEL_PATH = "model/harvest_model.h5"
TFLITE_MODEL_PATH = "model/harvest_model.tflite"
DATA_DIR = "data"  # 学習データのディレクトリ

# 収穫状態のカテゴリ
CATEGORIES = ['pre_harvest', 'harvest', 'post_harvest']

# 学習済みモデルの読み込み関数
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        print("既存のモデルを読み込んでいます...")
        return tf.keras.models.load_model(MODEL_PATH)

    print("新しいモデルを学習します...")
    images, labels = load_images(DATA_DIR)
    images = images / 255.0

    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(CATEGORIES), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    os.makedirs('model', exist_ok=True)
    model.save(MODEL_PATH)

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

    convert_model_to_tflite(model)
    return model

# 画像データの読み込み
def load_images(data_dir):
    images, labels = [], []
    for category in CATEGORIES:
        category_path = os.path.join(data_dir, category)
        label = CATEGORIES.index(category)
        for img_name in os.listdir(category_path):
            img_path = os.path.join(category_path, img_name)
            img = image.load_img(img_path, target_size=(150, 150))
            images.append(image.img_to_array(img))
            labels.append(label)
    return np.array(images), np.array(labels)

# TensorFlowモデルをTFLiteに変換
def convert_model_to_tflite(model):
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(TFLITE_MODEL_PATH, 'wb') as f:
            f.write(tflite_model)
        print(f"TensorFlow Liteモデルが保存されました: {TFLITE_MODEL_PATH}")
    except Exception as e:
        print(f"モデル変換エラー: {e}")

# 画像をキャプチャして予測
def capture_image_and_predict(model):
    picam2 = Picamera2()
    picam2.start()
    time.sleep(2)  # カメラ起動待機

    picam2.capture_file("captured_image.jpg")
    picam2.stop()

    img = image.load_img("captured_image.jpg", target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    result = CATEGORIES[predicted_class]

    print(f"予測された収穫状態: {result}")
    return result

# シリアル通信でArduinoと連携
def serial_communication(model):
    print("Arduino からのリクエストを待機中...")

    while True:
        if ser.in_waiting:
            message = ser.readline().decode('utf-8').strip()
            print(f"Arduinoから受信: {message}")

            if message == "AI Judgment Request":
                print("収穫判定を開始します...")
                result = capture_image_and_predict(model)

                # 判定結果をArduinoに送信
                ser.write((result + "\n").encode('utf-8'))
                print(f"Arduinoへ送信: {result}")

# メイン処理
def main():
    model = load_or_train_model()
    serial_communication(model)

if __name__ == '__main__':
    main()
