import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Could not open webcam.")

while True:
    r, frame = cap.read()

    if not r:
        print("Failed to read frame.")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("p"):
        break

cap.release()
cv2.destroyAllWindows()