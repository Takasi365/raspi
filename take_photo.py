import libcamera
import time

# カメラを初期化
camera = libcamera.Camera()

# カメラ設定
camera.start()

# 写真を撮影して保存
camera.capture_file('test_image.jpg')

# 撮影後にカメラを停止
camera.stop()

print("写真を撮影しました")
