from picamera2 import Picamera2
import time

# カメラのセットアップ
picam2 = Picamera2()
picam2.start()

# 画像キャプチャの関数
def capture_image():
    try:
        # 画像をキャプチャして保存
        picam2.capture_file("captured_image.jpg")
        print("画像をキャプチャしました: captured_image.jpg")
    except Exception as e:
        print(f"画像キャプチャ中にエラーが発生しました: {e}")

# 画像キャプチャを実行
capture_image()

# カメラの停止
picam2.stop()
