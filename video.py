import cv2
from ultralytics import YOLO
import time

# 1. Load your best weights from the directory you just found
model = YOLO("/home/abouvel/winter2025/FPI_Project/train/weights/best.pt")

# 2. Open the webcam
cap = cv2.VideoCapture(-1)

phone_start_time = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO inference on the live frame
    results = model(frame, conf=0.5, verbose=False)
    
    distracted = False
    for r in results:
        # Check if 'cell phone' is in the detected classes
        # (Assuming 'cell phone' is one of your class names)
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            
            if label == 'cell phone':
                distracted = True

    # 3. Logic: If phone is seen for more than 3 seconds
    if distracted:
        if phone_start_time is None:
            phone_start_time = time.time()
        
        elapsed = time.time() - phone_start_time
        if elapsed > 3:
            print("🚨 GET OFF YOUR PHONE! 🚨")
            # This is where you'd trigger your motivating clip/noise
    else:
        phone_start_time = None

    # Show the live feed (optional for the final website)
    cv2.imshow("Strava for Studying - AI Monitor", results[0].plot())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()