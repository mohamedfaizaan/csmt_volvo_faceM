import cv2
import mediapipe as mp
import numpy as np
import math

# -----------------------------
# MediaPipe Face Landmarker
# -----------------------------

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

model_path = "models/face_landmarker.task"

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=model_path
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

# -----------------------------
# Function to calculate distance
# -----------------------------

def distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )


# -----------------------------
# Eye Aspect Ratio
# -----------------------------

def eye_aspect_ratio(landmarks, eye_points):

    # Eye height
    vertical_distance = distance(
        landmarks[eye_points[1]],
        landmarks[eye_points[5]]
    )

    vertical_distance_2 = distance(
        landmarks[eye_points[2]],
        landmarks[eye_points[4]]
    )

    # Eye width
    horizontal_distance = distance(
        landmarks[eye_points[0]],
        landmarks[eye_points[3]]
    )

    # Calculate EAR
    ear = (
        vertical_distance +
        vertical_distance_2
    ) / (2.0 * horizontal_distance)

    return ear


# -----------------------------
# Eye landmark points
# -----------------------------

# These are points around the eyes
LEFT_EYE = [33, 160, 158, 133, 153, 144]

RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# -----------------------------
# Open S22 Ultra camera
# -----------------------------

camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("Could not open camera.")
    exit()

print("Camera started!")
print("Eye detection started!")
print("Press Q to quit.")


# -----------------------------
# Start Face Landmarker
# -----------------------------

with FaceLandmarker.create_from_options(options) as landmarker:

    timestamp = 0

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not read camera frame.")
            break

        # BGR → RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Create MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.asarray(rgb_frame)
        )

        # Detect landmarks
        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        timestamp += 33

        # -----------------------------
        # If face detected
        # -----------------------------

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            # Calculate left eye EAR
            left_ear = eye_aspect_ratio(
                landmarks,
                LEFT_EYE
            )

            # Calculate right eye EAR
            right_ear = eye_aspect_ratio(
                landmarks,
                RIGHT_EYE
            )

            # Average both eyes
            average_ear = (
                left_ear + right_ear
            ) / 2

            # -----------------------------
            # Decide eye state
            # -----------------------------

            if average_ear < 0.20:
                eye_status = "EYES: CLOSED"
            else:
                eye_status = "EYES: OPEN"

            # -----------------------------
            # Show result
            # -----------------------------

            cv2.putText(
                frame,
                eye_status,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"EAR: {average_ear:.2f}",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # Show camera
        cv2.imshow(
            "Eye Detection",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# -----------------------------
# Cleanup
# -----------------------------

camera.release()
cv2.destroyAllWindows()
