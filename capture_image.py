import time
import picamera
import os

# 撮影した画像を保存するディレクトリの設定
save_dir = 'data/test_images'

# 保存先のディレクトリがなければ作成
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 撮影する画像のファイル名（現在時刻を基にしたファイル名）
image_filename = os.path.join(save_dir, 'test_image_' + time.strftime("%Y%m%d_%H%M%S") + '.jpg')

# カメラの設定
with picamera.PICamera() as camera:
    camera.resolution = (1280, 720)  # 解像度の設定
    camera.start_preview()  # プレビューを開始
    time.sleep(2)  # プレビューを2秒間表示（カメラの準備時間）

    # 画像を撮影して保存
    camera.capture(image_filename)
    print(f"Image saved as {image_filename}")

# プレビューを終了
camera.stop_preview()
