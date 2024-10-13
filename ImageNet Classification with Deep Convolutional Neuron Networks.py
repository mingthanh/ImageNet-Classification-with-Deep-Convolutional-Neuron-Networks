# tải các thư viện cần thiết
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# tải mô hình ResNet50 đã được huấn luyện trước trên ImageNet
model = ResNet50(weights='imagenet')

# định nghĩa hàm để tải và tiền xử lý hình ảnh từ URL
def load_and_preprocess_image(img_url):
    # thêm User-Agent để tránh bị từ chối bởi server
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(img_url, headers=headers)
        # kiểm tra xem yêu cầu có thành công không
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image from URL: {e}")
        return None, None

    # kiểm tra Content-Type để đảm bảo rằng URL trỏ tới hình ảnh
    content_type = response.headers.get('Content-Type')
    if 'image' not in content_type:
        print(f"URL does not point to an image. Content-Type: {content_type}")
        return None, None

    try:
        # tải hình ảnh sử dụng PIL
        img = image.load_img(BytesIO(response.content), target_size=(224, 224))
        # chuyển đổi hình ảnh thành mảng numpy
        img_array = image.img_to_array(img)
        # thêm một chiều mới để phù hợp với đầu vào của mô hình
        img_array_expanded = np.expand_dims(img_array, axis=0)
        # tiền xử lý hình ảnh theo yêu cầu của ResNet50
        img_preprocessed = preprocess_input(img_array_expanded)
        return img, img_preprocessed
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

# định nghĩa hàm để hiển thị hình ảnh và kết quả dự đoán
def display_prediction(original_img, decoded_preds):
    plt.imshow(original_img)
    plt.axis('off')
    plt.title("Dự đoán:")
    plt.show()

    for i, pred in enumerate(decoded_preds):
        print(f"{i+1}. {pred[1]} ({pred[2]*100:.2f}%)")

# tải và dự đoán một hình ảnh từ URL
def predict_image_from_url(img_url):
    original_img, preprocessed_img = load_and_preprocess_image(img_url)
    if original_img is not None and preprocessed_img is not None:
        # dự đoán lớp cho hình ảnh
        preds = model.predict(preprocessed_img)
        # giải mã các dự đoán thành các lớp có ý nghĩa
        decoded_preds = decode_predictions(preds, top=3)[0]
        # hiển thị hình ảnh và kết quả dự đoán
        display_prediction(original_img, decoded_preds)
    else:
        print("Không thể dự đoán hình ảnh vì có lỗi trong quá trình tải hoặc xử lý.")

# sử dụng hàm để dự đoán hình ảnh từ URL
img_urls = [
    'https://upload.wikimedia.org/wikipedia/commons/5/54/Golden_Retriever_medium-to-light-coat.jpg',  # Golden Retriever
    'https://upload.wikimedia.org/wikipedia/commons/9/99/Black_dog.jpg',                              # chú chó đen
    'https://upload.wikimedia.org/wikipedia/commons/3/3f/Fronalpstock_big.jpg',                       # cảnh thiên nhiên
    'https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png'        # ảnh PNG minh họa
]

for url in img_urls:
    print(f"\nĐang xử lý hình ảnh từ URL: {url}")
    predict_image_from_url(url)
    print("\n" + "-"*60 + "\n")
