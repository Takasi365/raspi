import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import serial
import time
from picamera2 import Picamera2
import subprocess

# シリアル通信設定（Arduinoのポートに合わせて変更）
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 9600
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# モデルの保存パス
MODEL_PATH = "model/harvest_model.h5"
CATEGORIES = ['pre_harvest', 'harvest', 'post_harvest']

# モデルの読み込み
def load_model():
    if os.path.exists(MODEL_PATH):
        print("既存のモデルを読み込んでいます...")
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        raise FileNotFoundError("モデルが見つかりません！事前に学習を行ってください。")

# 画像をキャプチャして収穫状態を判定
def capture_image_and_predict(model):
    try:
        # カメラの初期化
        picam2 = Picamera2()
        picam2.start()
        time.sleep(2)  # カメラ起動待機

        # 画像をキャプチャ
        image_path = "/home/takase/hervest_detection/data/captured_image.jpg"
        picam2.capture_file(image_path)
        
        # カメラを閉じる（リソース解放）
        picam2.stop()
        picam2.close()

        # 画像を読み込んで前処理
        img = image.load_img(image_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # 予測
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        result = CATEGORIES[predicted_class]

        print(f"予測された収穫状態: {result}")
        return result

    except Exception as e:
        print(f"カメラエラー: {e}")
        return "error"

# 画像をGitHubにアップロード
def upload_image_to_github(image_path):
    try:
        # 画像が保存されていることを確認
        if not os.path.exists(image_path):
            print(f"画像ファイル {image_path} が見つかりません")
            return

        # Gitの状態を確認して、アップロードできるようにする
        subprocess.run(["git", "add", image_path])

        # 変更をコミット
        subprocess.run(["git", "commit", "-m", "Add captured image"])

        # GitHubにプッシュ
        subprocess.run(["git", "push", "origin", "main"])
        print("GitHubに画像をアップロードしました。")

    except Exception as e:
        print(f"GitHubへのアップロードエラー: {e}")

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

                # 画像のアップロード
                image_path = "/home/takase/hervest_detection/data/captured_image.jpg"
                upload_image_to_github(image_path)

                # 判定結果をArduinoに送信
                ser.write((result + "\n").encode('utf-8'))
                print(f"Arduinoへ送信: {result}")

# メイン処理
def main():
    model = load_model()
    serial_communication(model)

if __name__ == '__main__':
    main()
