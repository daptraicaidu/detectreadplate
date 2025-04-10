from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64, cv2, numpy as np, os, shutil
from sklearn.cluster import KMeans
from ultralytics import YOLO
# import uvicorn

app = FastAPI()

# Cho phép CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Danh sách nhãn ký tự
class_names = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Biến model toàn cục
model = None
models = None

# Load model khi server khởi động
@app.on_event("startup")
def load_models():
    global model, models
    print("🚀 Đang load model YOLO...")
    if not os.path.exists("bestDetect.pt") or not os.path.exists("bestRead.pt"):
        raise FileNotFoundError("❌ Thiếu file bestDetect.pt hoặc bestRead.pt trong thư mục.")
    model = YOLO("bestDetect.pt")
    models = YOLO("bestRead.pt")
    print("✅ Load model thành công.")

# Route root check server sống
@app.get("/")
def root():
    return {"status": "Server đang chạy ngon lành 😎"}

# API detect biển số
@app.post("/detect")
async def detect_license_plate(file: UploadFile = File(...)):
    contents = await file.read()
    print(f"📥 Nhận file: {file.filename}, size: {len(contents)} bytes")

    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        print("❌ Ảnh decode fail.")
        return JSONResponse(content={"error": "Không thể đọc ảnh"}, status_code=400)

    print("🧠 Đang chạy YOLO detect...")
    try:
        results = model(img)
    except Exception as e:
        print("🔥 Lỗi khi detect:", e)
        return JSONResponse(content={"error": "Lỗi khi chạy model detect"}, status_code=500)

    plate_images_base64 = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            license_plate = img[y1:y2, x1:x2]
            if license_plate.size == 0:
                continue
            _, buffer = cv2.imencode(".jpg", license_plate)
            plate_base64 = base64.b64encode(buffer).decode("utf-8")
            plate_images_base64.append(plate_base64)

    if not plate_images_base64:
        print("⚠️ Không tìm thấy biển số nào.")
        return JSONResponse(content={"message": "Không phát hiện biển số nào"}, status_code=404)

    print(f"✅ Phát hiện {len(plate_images_base64)} biển số.")
    return {"plates": plate_images_base64}

# API đọc biển số từ ảnh cắt
@app.post("/read")
async def recognize_plate(file: UploadFile = File(...)):
    temp_file = f"temp_{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image = cv2.imread(temp_file)
    os.remove(temp_file)

    if image is None:
        print("❌ Ảnh không đọc được khi đọc biển.")
        return JSONResponse(content={"error": "Không thể đọc ảnh"}, status_code=400)

    print("🔍 Đang đọc ký tự từ ảnh...")
    try:
        results = models(image)
    except Exception as e:
        print("🔥 Lỗi khi đọc ký tự:", e)
        return JSONResponse(content={"error": "Lỗi khi chạy model đọc"}, status_code=500)

    detected_chars = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            label = class_names[class_id]
            detected_chars.append({"char": label, "x": x1, "y": y1})

    if not detected_chars:
        print("⚠️ Không nhận diện được ký tự nào.")
        return {"message": "Không nhận diện được biển số!"}

    y_coords = np.array([char["y"] for char in detected_chars]).reshape(-1, 1)

    if len(detected_chars) <= 2:
        sorted_chars = sorted(detected_chars, key=lambda char: char["x"])
        plate_text = "".join([char["char"] for char in sorted_chars])
        return {"plate_text": plate_text}

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(y_coords)
    labels = kmeans.labels_

    group1 = [char for i, char in enumerate(detected_chars) if labels[i] == 0]
    group2 = [char for i, char in enumerate(detected_chars) if labels[i] == 1]

    if np.mean([char["y"] for char in group1]) < np.mean([char["y"] for char in group2]):
        upper_row, lower_row = group1, group2
    else:
        upper_row, lower_row = group2, group1

    if abs(np.mean([char["y"] for char in upper_row]) - np.mean([char["y"] for char in lower_row])) < 10:
        upper_row.extend(lower_row)
        lower_row = []

    upper_row.sort(key=lambda char: char["x"])
    lower_row.sort(key=lambda char: char["x"])

    plate_text = "".join([char["char"] for char in upper_row])
    if lower_row:
        plate_text += "".join([char["char"] for char in lower_row])

    print("✅ Kết quả nhận diện:", plate_text)
    return {"plate_text": plate_text}



# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)

