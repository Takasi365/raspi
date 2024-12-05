from picamera2 import Picamera2
import time

# Picamera2のインスタンスを作成
picam2 = Picamera2()

# カメラの設定を開始
picam2.start()

# 画像を撮影して保存
picam2.capture_file('test_image.jpg')

# 5秒待機（撮影時間）
time.sleep(5)

# カメラを停止
picam2.stop()

print("画像の撮影が完了しました。")
