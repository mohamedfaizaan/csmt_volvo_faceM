import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Face Detector
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
RunningMode = mp.tasks.vision.RunningMode

# Location of our AI model
model_path = "models/blaze_face_short_range.tflite"

# Face detector settings
options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path=model_path
    ),
    running_mode=RunningMode.VIDEO,
    min_detection_confidence=0.5
)

# Open your S22 Ultra camera
camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("Could not open camera.")
    exit()

print("Camera started!")
print("Face detection started!")
print("Press Q to quit.")

# Start face detector
with FaceDetector.create_from_options(options) as detector:

    frame_timestamp = 0

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        # OpenCV uses BGR.
        # MediaPipe expects RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert OpenCV image to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.asarray(rgb_frame)
        )

        # Detect faces
        result = detector.detect_for_video(
            mp_image,
            frame_timestamp
        )

        # Increase timestamp for every frame
        frame_timestamp += 33

        # Draw boxes around detected faces
        for detection in result.detections:

            bbox = detection.bounding_box

            x = bbox.origin_x
            y = bbox.origin_y
            width = bbox.width
            height = bbox.height

            # Draw face box
            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2
            )

            # Face confidence
            confidence = detection.categories[0].score

            text = f"FACE {confidence * 100:.1f}%"

            cv2.putText(
                frame,
                text,
                (x, max(30, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # Show camera
        cv2.imshow("Face AI", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
