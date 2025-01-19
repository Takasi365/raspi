import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import serial
import time
from picamera2 import Picamera2
from flask import Flask, request, jsonify
import threading

# シリアル通信設定（Arduinoのポートに合わせて変更）
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 9600
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# モデルの保存パス
MODEL_PATH = "model/harvest_model.h5"
CATEGORIES = ['pre_harvest', 'harvest', 'post_harvest']

# Flask アプリの作成
app = Flask(__name__)

# モデルの読み込み
def load_model():
    if os.path.exists(MODEL_PATH):
        print("既存のモデルを読み込んでいます...")
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        raise FileNotFoundError("モデルが見つかりません！事前に学習を行ってください。")

# 画像をキャプチャして収穫状態を判定
def capture_image_and_predict():
    try:
        model = load_model()

        # カメラの初期化
        picam2 = Picamera2()
        picam2.start()
        time.sleep(2)  # カメラ起動待機

        # 画像をキャプチャ
        image_path = "/home/takase/hervest_detection/data/captured_image.jpg"  # 保存先を変更
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

# Arduino とのシリアル通信を処理
def serial_communication():
    print("Arduino からのリクエストを待機中...")

    while True:
        if ser.in_waiting:
            message = ser.readline().decode('utf-8').strip()
            print(f"Arduinoから受信: {message}")

            if message == "AI Judgment Request":
                print("収穫判定を開始します...")
                result = capture_image_and_predict()

                # 判定結果をArduinoに送信
                ser.write((result + "\n").encode('utf-8'))
                print(f"Arduinoへ送信: {result}")

# REST API エンドポイント（収穫判定リクエスト）
@app.route('/predict', methods=['POST'])
def predict():
    print("Unityから収穫判定リクエストを受信しました。")

    # 収穫判定を実行
    result = capture_image_and_predict()

    # 結果を JSON で返す
    return jsonify({'result': result})

# Flask サーバーの起動
def start_server():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# メイン処理
def main():
    # シリアル通信スレッドの開始
    serial_thread = threading.Thread(target=serial_communication, daemon=True)
    serial_thread.start()

    # Flask サーバーの開始
    start_server()

if __name__ == '__main__':
    main()
