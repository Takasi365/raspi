import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import serial
import time
from picamera2 import Picamera2
import subprocess
from flask import Flask, request, jsonify

# Flask アプリの初期化
app = Flask(__name__)

# シリアル通信設定（Arduinoのポートに合わせて変更）
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 9600
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# モデルの保存パス
MODEL_PATH = "model/harvest_model.h5"
CATEGORIES = ['pre_harvest', 'harvest', 'post_harvest']

# UR_A の定義（整数型）
UR_A = 0  # 初期値を設定

# モデルの読み込み
def load_model():
    if os.path.exists(MODEL_PATH):
        print("既存のモデルを読み込んでいます...")
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        raise FileNotFoundError("モデルが見つかりません！事前に学習を行ってください。")

model = load_model()

# 画像をキャプチャして収穫状態を判定
def capture_image_and_predict():
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
        if not os.path.exists(image_path):
            print(f"画像ファイル {image_path} が見つかりません")
            return

        print("Gitステータスを確認中...")
        result = subprocess.run(["git", "status"], capture_output=True, text=True, check=True)
        print(result.stdout)

        subprocess.run(["git", "add", image_path], check=True)
        subprocess.run(["git", "commit", "-m", "Add captured image"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)

        print("GitHubに画像をアップロードしました。")

    except subprocess.CalledProcessError as e:
        print(f"Gitコマンドエラー: {e.stderr}")
    except Exception as e:
        print(f"GitHubへのアップロードエラー: {e}")

# AI 判定リクエスト API
@app.route('/predict', methods=['POST'])
def predict():
    global UR_A

    print("収穫判定を開始します...")
    result = capture_image_and_predict()

    # 画像のアップロード
    image_path = "/home/takase/hervest_detection/data/captured_image.jpg"
    upload_image_to_github(image_path)

    # UR_A を更新
    UR_A = 1  # 収穫判定が完了したことを示す
    print(f"UR_A updated: {UR_A}")

    # Arduino に送信
    ser.write((result + "\n").encode('utf-8'))
    print(f"Arduinoへ送信: {result}")

    return jsonify({'status': 'success', 'result': result})

# UR_A の状態を取得する API
@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({'UR_A': UR_A})

# UR_A のリセット API
@app.route('/reset', methods=['POST'])
def reset_status():
    global UR_A
    UR_A = 0
    print(f"UR_A reset: {UR_A}")
    return jsonify({'status': 'success', 'UR_A': UR_A})

# メイン処理
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
