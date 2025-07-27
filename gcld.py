import cv2

rtsp_url = "rtsp://167.71.224.73:8554/mystream"

cap = cv2.VideoCapture(rtsp_url)

ret, frame = cap.read()

if ret:
    cv2.imwrite("frame.jpg", frame)  # save the frame as an image
    print("Frame captured and saved as frame.jpg")
else:
    print("Failed to capture frame")

cap.release()