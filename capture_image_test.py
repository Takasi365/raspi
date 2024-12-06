from picamera2 import Picamera2
import time

# カメラを初期化
picam2 = Picamera2()
picam2.start()

# 画像キャプチャ
picam2.capture_file("test_image.jpg")
print("画像をキャプチャしました")

# カメラを停止
picam2.stop()
