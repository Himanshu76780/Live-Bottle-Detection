import cv2
from ultralytics import YOLO


model = YOLO("best.pt")

cap = cv2.VideoCapture(0)

threshold = 0.4

while True:
   
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold and results.names[int(class_id)].lower() == "bottle":
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Bottle: {score:.2f}", (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Bottle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()