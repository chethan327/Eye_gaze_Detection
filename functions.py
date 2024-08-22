
import cv2 
import numpy as np
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                   min_detection_confidence=0.5,min_tracking_confidence = 0.5)
mp_drawing = mp.solutions.drawing_utils


# Eye landmark indices (based on MediaPipe Face Mesh landmarks)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

RIGHT_IRIS=[474,475,476,477]
LEFT_IRIS=[469,470,471,472]

L_H_LEFT=[33]
L_H_RIGHT=[133]
R_H_LEFT=[362]
R_H_RIGHT=[263]

# Threshold for EAR to determine if eye is closed
EAR_THRESHOLD = 0.25


def euclidean_distance(point1, point2):
    x1,y1=point1.ravel()
    x2,y2=point2.ravel()
    distance=math.sqrt((x2-x1)**2 +(y2-y1)**2)
    return distance

def iris_position(iris_center, right_point,left_point):
    center_to_right_dist=euclidean_distance(iris_center,right_point)
    total_distance=euclidean_distance(right_point,left_point)
    ratio=center_to_right_dist/total_distance
    iris_position=""
    if ratio<=0.42:
        iris_position="right"
    elif ratio >0.42 and ratio<=0.57:
        iris_position="center"
    else:
        iris_position="left"
        
    return iris_position ,ratio


# Define function to calculate EAR
def calculate_ear(eye_landmarks):
    # Calculate distances between vertical eye landmarks
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    # Calculate distance between horizontal eye landmarks
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    # Compute EAR
    ear = (A + B) / (2.0 * C)
    print(f"ear : {ear}")
    return ear



# Define function to get eye landmarks
def get_eye_landmarks(landmarks, indices):
    return np.array([(landmarks[idx].x, landmarks[idx].y) for idx in indices])
