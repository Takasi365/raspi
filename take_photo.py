from picamera2 import Picamera2

# Picamera2 のインスタンスを作成
picam2 = Picamera2()

# カメラの設定（必要に応じて設定）
picam2.configure(picam2.create_still_configuration())

# 写真を撮影して保存
picam2.start()
picam2.capture_file("test.jpg")
picam2.stop()

print("写真を撮影しました")
