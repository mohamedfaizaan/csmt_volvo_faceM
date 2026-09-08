import cv2

camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("Could not open Camera 1")
    exit()

# Ask the camera for MJPEG video
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Camera 1 connected!")

while True:
    success, frame = camera.read()

    if not success:
        print("Could not receive video")
        break

    cv2.imshow("S22 Ultra Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
