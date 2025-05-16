#code train

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Đường dẫn dữ liệu
train_dir = "/kaggle/input/datafood"
img_width, img_height = 224, 224

# Tham số huấn luyện
batch_size = 32
learning_rate = 0.0001
epochs = 100

# Tạo ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    brightness_range=[0.1, 1.5],
    fill_mode='nearest',
    validation_split=0.1
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Lưu nhãn
labels_dict = train_generator.class_indices
labels = [None] * len(labels_dict)
for label, index in labels_dict.items():
    labels[index] = label

labels_path = "/kaggle/working/Monan2.npy"
np.save(labels_path, labels)

print(f"Nhãn đã được lưu tại: {labels_path}")
print(f"Danh sách nhãn: {labels}")
print(f"img_width: {img_width}, img_height: {img_height}")

# Xây dựng mô hình
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(len(train_generator.class_indices), activation="softmax")
])

# Compile mô hình
model.compile(optimizer=tf.keras.optimizers.AdamW(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Callbacks (chỉ EarlyStopping để đơn giản hóa, bạn có thể thêm ModelCheckpoint nếu muốn)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Huấn luyện mô hình
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# Chuyển đổi và lưu mô hình dưới dạng TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_model_path = "/kaggle/working/nhandienCNN.tflite"
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"\nMô hình đã được huấn luyện và lưu dưới dạng TFLite tại: {tflite_model_path}")

# (Tùy chọn) Hiển thị đồ thị lịch sử huấn luyện
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()







#code nhan dien
import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
import os


file_model_path = "/content/drive/MyDrive/nhandienCNN.tflite" 



yolo_model = YOLO("yolov8n.pt")


interpreter = tf.lite.Interpreter(model_path=file_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']


class_names = [
    "Ca hu kho", "Canh cai", "Canh chua", "Com trang", "Dau hu sot ca",
    "Ga chien", "Rau muong xao", "Thit kho", "Thit kho trung", "Trung chien"
]

food_prices = {
    "Ca hu kho": 20000,
    "Canh cai": 10000,
    "Canh chua": 10000,
    "Com trang": 5000,
    "Dau hu sot ca": 20000,
    "Ga chien": 20000,
    "Rau muong xao": 10000,
    "Thit kho": 20000,
    "Thit kho trung": 20000,
    "Trung chien": 5000
}

def classify_image(image):
    results = yolo_model(image)
    detections = results[0].boxes.data.cpu().numpy()
    predicted_foods = []
    total_price = 0
    for det in detections:
        x1, y1, x2, y2, confidence, class_id = det
        class_id = int(class_id)

        if confidence < 0.3:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        resized = cv2.resize(crop, (input_shape[1], input_shape[2]))
        input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_index = np.argmax(output_data)
        predicted_food = class_names[predicted_index]

        predicted_foods.append(predicted_food)
        total_price += food_prices.get(predicted_food, 0)

    if not predicted_foods:
        return "Không phát hiện được món ăn", "0đ"

    result_text = "\n".join([f"{food}: {food_prices[food]:,}đ" for food in predicted_foods])
    total_price_text = f"{total_price:,}đ"
    return result_text, total_price_text

with gr.Blocks(title="Nhận diện Món Ăn Canteen UEH", theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1 style='text-align: center;'>🍜 Nhận diện Món Ăn Canteen UEH (AI Challenges 3ITECH 2025)</h1>")
    gr.Markdown(f"<h3 style='text-align: center;'>Các món ăn: {', '.join(class_names)}</h3>")

    with gr.Row():
        with gr.Column():
            gr.Markdown("📸 Tải lên ảnh mâm cơm:")
            image_input = gr.Image(type="numpy", label="Ảnh mâm cơm", interactive=True)
            nhan_dien_button = gr.Button("🔍 Nhận diện và Tính tiền", variant="primary")
            reload_button = gr.Button("🔄 Tải lại", variant="secondary")

        with gr.Column():
            gr.Markdown("🌼 Kết quả nhận diện:")
            food_output = gr.Textbox(label="Các món ăn đã phát hiện", lines=5)
            total_output = gr.Textbox(label="Tổng tiền", lines=1)

    nhan_dien_button.click(fn=classify_image, inputs=image_input, outputs=[food_output, total_output])
    reload_button.click(fn=lambda: (None, "", ""), outputs=[
        image_input, food_output, total_output
    ])

demo.launch(debug=True, share=True)
