from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from fastapi.responses import JSONResponse
import os
import shutil
import uvicorn
from sklearn.cluster import KMeans




app = FastAPI()

# Thêm middleware CORS để cho phép yêu cầu từ mọi nguồn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các domain (hoặc bạn có thể chỉ định một số domain cụ thể)
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức HTTP
    allow_headers=["*"],  # Cho phép tất cả các headers
)

# Load mô hình YOLOv8 nhận diện biển số
model = YOLO('bestDetect.pt')
# Load mô hình YOLOv8 đọc chữ
models = YOLO("bestRead.pt")

# Danh sách nhãn ký tự
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 
               'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
               'W', 'X', 'Y', 'Z']


@app.post("/detect/")
async def detect_license_plate(file: UploadFile = File(...)):
    # Đọc ảnh từ file tải lên vào bộ nhớ
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(content={"error": "Không thể đọc ảnh"}, status_code=400)

    results = model(img)  # Nhận diện biển số

    plate_images_base64 = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            license_plate = img[y1:y2, x1:x2]  # Cắt biển số từ ảnh gốc

            if license_plate.size == 0:
                continue  # Bỏ qua nếu không có nội dung

            # Chuyển ảnh thành chuỗi Base64
            _, buffer = cv2.imencode(".jpg", license_plate)
            plate_base64 = base64.b64encode(buffer).decode("utf-8")
            plate_images_base64.append(plate_base64)

    if not plate_images_base64:
        return JSONResponse(content={"message": "Không phát hiện biển số nào"}, status_code=404)

    return {"plates": plate_images_base64}

@app.post("/read/")
async def recognize_plate(file: UploadFile = File(...)):
    # Lưu file ảnh tạm thời
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Đọc ảnh
    image = cv2.imread(temp_file)
    os.remove(temp_file)  # Xóa file ngay sau khi đọc để tránh tốn bộ nhớ
    
    # Nhận diện biển số
    results = models(image)
    detected_chars = []
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = class_names[class_id]
            detected_chars.append({"char": label, "x": x1, "y": y1})

    if not detected_chars:
        return {"message": "Không nhận diện được biển số!"}

    # Lấy danh sách tọa độ y
    y_coords = np.array([char["y"] for char in detected_chars]).reshape(-1, 1)

    # Nếu số ký tự quá ít, không cần chia dòng
    if len(detected_chars) <= 2:
        sorted_chars = sorted(detected_chars, key=lambda char: char["x"])
        plate_text = "".join([char["char"] for char in sorted_chars])
        return {"plate_text": plate_text}

    # Áp dụng K-Means clustering để chia thành 2 dòng
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(y_coords)
    
    # Lấy nhãn của từng ký tự
    labels = kmeans.labels_

    # Phân chia thành 2 dòng
    group1 = [char for i, char in enumerate(detected_chars) if labels[i] == 0]
    group2 = [char for i, char in enumerate(detected_chars) if labels[i] == 1]

    # Xác định dòng trên và dòng dưới dựa vào tọa độ y trung bình
    if np.mean([char["y"] for char in group1]) < np.mean([char["y"] for char in group2]):
        upper_row, lower_row = group1, group2
    else:
        upper_row, lower_row = group2, group1

    # Nếu khoảng cách giữa 2 dòng quá nhỏ, gộp lại thành 1 dòng duy nhất
    if abs(np.mean([char["y"] for char in upper_row]) - np.mean([char["y"] for char in lower_row])) < 10:
        upper_row.extend(lower_row)
        lower_row = []

    # Sắp xếp từng dòng theo tọa độ x (trái → phải)
    upper_row.sort(key=lambda char: char["x"])
    lower_row.sort(key=lambda char: char["x"])

    # Kết hợp biển số: Dòng trên trước, dòng dưới sau
    plate_text = "".join([char["char"] for char in upper_row])
    if lower_row:
        plate_text += "".join([char["char"] for char in lower_row])

    return {"plate_text": plate_text}


#uvicorn api:app --reload
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

