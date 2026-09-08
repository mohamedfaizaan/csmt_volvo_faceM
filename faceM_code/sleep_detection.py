import cv2
import mediapipe as mp
import numpy as np
import math
import time

# ==========================================
# MEDIAPIPE FACE LANDMARKER
# ==========================================

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


# ==========================================
# DISTANCE BETWEEN TWO FACE POINTS
# ==========================================

def distance(point1, point2):
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )


# ==========================================
# CALCULATE EYE ASPECT RATIO
# ==========================================

def eye_aspect_ratio(landmarks, eye_points):

    vertical_1 = distance(
        landmarks[eye_points[1]],
        landmarks[eye_points[5]]
    )

    vertical_2 = distance(
        landmarks[eye_points[2]],
        landmarks[eye_points[4]]
    )

    horizontal = distance(
        landmarks[eye_points[0]],
        landmarks[eye_points[3]]
    )

    ear = (vertical_1 + vertical_2) / (2 * horizontal)

    return ear


# ==========================================
# EYE LANDMARK POINTS
# ==========================================

LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]

RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]


# ==========================================
# OPEN CAMERA
# ==========================================

camera = cv2.VideoCapture(1)

if not camera.isOpened():

    print("Could not open camera.")

    exit()


print("Camera started!")
print("Sleep detection started!")
print("Press Q to quit.")


# ==========================================
# VARIABLES
# ==========================================

eyes_closed = False

closed_start_time = None


# ==========================================
# START MEDIAPIPE
# ==========================================

with FaceLandmarker.create_from_options(options) as landmarker:

    timestamp = 0

    while True:

        # ----------------------------------
        # READ CAMERA
        # ----------------------------------

        success, frame = camera.read()

        if not success:

            print("Could not read camera frame.")

            break


        # ----------------------------------
        # CONVERT BGR → RGB
        # ----------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------
        # CREATE MEDIAPIPE IMAGE
        # ----------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.asarray(rgb_frame)
        )


        # ----------------------------------
        # DETECT FACE
        # ----------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        timestamp += 33


        # ==================================
        # IF FACE FOUND
        # ==================================

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]


            # ----------------------------------
            # CALCULATE EAR
            # ----------------------------------

            left_ear = eye_aspect_ratio(
                landmarks,
                LEFT_EYE
            )

            right_ear = eye_aspect_ratio(
                landmarks,
                RIGHT_EYE
            )

            average_ear = (
                left_ear + right_ear
            ) / 2


            # ----------------------------------
            # CHECK EYES
            # ----------------------------------

            if average_ear < 0.20:

                # Eyes have just closed
                if not eyes_closed:

                    eyes_closed = True

                    closed_start_time = time.time()


                # Calculate how long eyes
                # have been closed
                closed_duration = (
                    time.time() - closed_start_time
                )


            else:

                # Eyes opened again
                eyes_closed = False

                closed_start_time = None

                closed_duration = 0


            # ==================================
            # DETERMINE STATUS
            # ==================================

            if eyes_closed:

                if closed_duration < 1:

                    status = "BLINKING"

                elif closed_duration < 3:

                    status = "DROWSY"

                else:

                    status = "SLEEPING"

            else:

                status = "AWAKE"


            # ==================================
            # DISPLAY EYE STATUS
            # ==================================

            if eyes_closed:

                eye_text = "EYES: CLOSED"

            else:

                eye_text = "EYES: OPEN"


            cv2.putText(
                frame,
                eye_text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


            # ==================================
            # DISPLAY EAR
            # ==================================

            cv2.putText(
                frame,
                f"EAR: {average_ear:.2f}",
                (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


            # ==================================
            # DISPLAY CLOSED TIME
            # ==================================

            if eyes_closed:

                cv2.putText(
                    frame,
                    f"Closed: {closed_duration:.1f} sec",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )


            # ==================================
            # DISPLAY FINAL STATUS
            # ==================================

            cv2.putText(
                frame,
                f"STATUS: {status}",
                (30, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                3
            )


        else:

            # No face detected
            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )


        # ==================================
        # SHOW CAMERA
        # ==================================

        cv2.imshow(
            "Sleep Detection",
            frame
        )


        # ==================================
        # QUIT
        # ==================================

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# ==========================================
# CLEANUP
# ==========================================

camera.release()

cv2.destroyAllWindows()
