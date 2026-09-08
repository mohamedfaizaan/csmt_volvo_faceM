import cv2
import mediapipe as mp
import numpy as np

# -----------------------------
# MediaPipe Face Landmarker
# -----------------------------

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Our face landmark model
model_path = "models/face_landmarker.task"

# Face Landmarker settings
options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=model_path
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

# -----------------------------
# Open S22 Ultra camera
# -----------------------------

camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("Could not open camera.")
    exit()

print("Camera started!")
print("Face landmarks started!")
print("Press Q to quit.")

# -----------------------------
# Start Face Landmarker
# -----------------------------

with FaceLandmarker.create_from_options(options) as landmarker:

    timestamp = 0

    while True:

        # Get camera frame
        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        # OpenCV BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.asarray(rgb_frame)
        )

        # Detect face landmarks
        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        timestamp += 33

        # -----------------------------
        # Draw landmarks
        # -----------------------------

        if result.face_landmarks:

            for face_landmarks in result.face_landmarks:

                height, width, _ = frame.shape

                for landmark in face_landmarks:

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    # Draw a small point
                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )

        # Show camera
        cv2.imshow(
            "Face Landmarks",
            frame
        )

        # Q = quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Cleanup
camera.release()
cv2.destroyAllWindows()
