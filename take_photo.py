from picamera2 import Picamera2
from time import sleep

# Picamera2インスタンスを作成
picam2 = Picamera2()

# カメラを起動
picam2.start()
sleep(2)  # カメラの準備待機時間

# 写真を撮影
picam2.capture_file("test.jpg")
print("Photo taken: test.jpg")
