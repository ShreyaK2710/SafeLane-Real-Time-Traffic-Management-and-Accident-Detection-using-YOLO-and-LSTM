import cv2
from ultralytics import YOLO

model = YOLO("yolov8x.pt") 


video_path = r"C:\Users\SHREYA\Downloads\test1.mp4"
cap = cv2.VideoCapture(video_path)

cv2.namedWindow("YOLO Traffic Detection", cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  

    results = model(frame)

    vehicle_count = 0
    for box in results[0].boxes:
        cls = int(box.cls[0].item())  
        class_name = model.names[cls]  

        if class_name in ["car", "bus", "truck", "motorcycle"]:
            vehicle_count += 1  

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()
        label = f"{class_name} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    cv2.imshow("YOLO Traffic Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
