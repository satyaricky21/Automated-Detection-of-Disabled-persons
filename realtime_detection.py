from ultralytics import YOLO
import cv2

MODEL_PATH = r"D:\robotrix\disabled detection\runs\detect\train\weights\best.pt"
CONFIDENCE = 0.25

# Allowed classes ONLY (no normal person)
ALLOWED_CLASSES = {
    "person_wheelchair",
    "person_walking_frame",
    "person_crutches",
    "person_cane",
    "stroller",
    "bicycle"
}

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONFIDENCE, device=0)

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        # 🚫 Skip unwanted detections
        if label not in ALLOWED_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label.upper(), (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Mobility Aid Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
