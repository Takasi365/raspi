import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

# ディレクトリパスの設定
BASE_DIR = "../harvest_prediction"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "harvest_model.h5")

# 学習用データジェネレータの設定
def create_data_generators(data_dir, img_size=(128, 128), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        validation_split=0.2
    )
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    val_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    return train_generator, val_generator

# モデル構築
def build_model(input_shape=(128, 128, 3), num_classes=3):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# モデルの学習
def train_model():
    train_gen, val_gen = create_data_generators(DATA_DIR)
    model = build_model()
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    print("Model saved to", MODEL_PATH)

# モデルをロードして予測
def predict_image(image_path):
    model = tf.keras.models.load_model(MODEL_PATH)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    labels = ["Pre-Harvest", "Harvest", "Post-Harvest"]
    print("Prediction:", labels[np.argmax(prediction)])

# メイン実行部
if __name__ == "__main__":
    mode = input("Choose mode: 'train' or 'predict': ").strip().lower()
    if mode == 'train':
        train_model()
    elif mode == 'predict':
        image_path = input("Enter image path: ").strip()
        predict_image(image_path)
    else:
        print("Invalid mode. Choose 'train' or 'predict'.")
