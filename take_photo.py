from picamera2 import Picamera2
import time

# Picamera2オブジェクトを初期化
picam2 = Picamera2()

# カメラ設定を初期化
picam2.start()

# 写真を撮影して保存
output_file = "photo.jpg"
picam2.capture_file(output_file)

print(f"写真が撮影され、{output_file}に保存されました。")

# 終了
picam2.close()
