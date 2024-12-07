from picamera2 import Picamera2
import time

# Picamera2の初期化
picam2 = Picamera2()

# カメラを起動
picam2.start()

# 撮影する前に少し待機
time.sleep(2)

# 画像をキャプチャし、ファイルに保存
picam2.capture_file("test_image.jpg")

# 撮影完了のメッセージ
print("画像が撮影され、'test_image.jpg' として保存されました。")

# カメラを停止
picam2.stop()
