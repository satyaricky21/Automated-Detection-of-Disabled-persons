import cv2
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO
import sys
import os

# ================= CONFIG =================
CONFIDENCE = 0.4
WINDOW_NAME = "Disabled Person Detection"
# =========================================

# ---------- FIX FOR EXE MODEL PATH ----------
def get_model_path():
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, 'best.pt')
    return "best.pt"

MODEL_PATH = get_model_path()
# -------------------------------------------

CLASS_NAMES = {
    0: "Bicycle",
    1: "Person with Cane",
    2: "Person with Crutches",
    3: "Person with Walking Frame",
    4: "Person with Wheelchair",
    5: "Stroller"
}

# Load model
model = YOLO(MODEL_PATH)


# ---------- IMAGE DETECTION ----------
def detect_from_image():
    while True:
        root = tk.Tk()
        root.withdraw()

        img_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
        )

        if not img_path:
            print("Image selection cancelled.")
            break

        img = cv2.imread(img_path)
        results = model(img, conf=CONFIDENCE)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls not in CLASS_NAMES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{CLASS_NAMES[cls]} {box.conf[0]:.2f}"

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Image Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        again = input("Upload another image? (y/n): ").lower()
        if again != "y":
            break


# ---------- CAMERA DETECTION ----------
def detect_from_camera(cam_index=0):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("❌ Camera not accessible")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 900, 600)

    print("▶ Webcam started")
    print("Press Q or ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break

        results = model(frame, conf=CONFIDENCE)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls not in CLASS_NAMES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{CLASS_NAMES[cls]} {box.conf[0]:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        # EXIT KEYS
        if key == ord('q') or key == ord('Q') or key == 27:
            print("✅ Webcam closed")
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------- MAIN MENU ----------
def main():
    while True:
        print("\n==== Disabled Detection System ====")
        print("1. Default Webcam")
        print("2. External Webcam")
        print("3. Select Image")
        print("4. Exit")

        choice = input("Select option (1/2/3/4): ")

        if choice == "1":
            detect_from_camera(0)

        elif choice == "2":
            try:
                cam_id = int(input("Enter external webcam index (1/2): "))
                detect_from_camera(cam_id)
            except ValueError:
                print("❌ Invalid camera index")

        elif choice == "3":
            detect_from_image()

        elif choice == "4":
            print("Exiting system.")
            break

        else:
            print("❌ Invalid option")


if __name__ == "__main__":
    main()