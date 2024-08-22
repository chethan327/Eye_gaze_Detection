from functions import *

import cv2
import mediapipe as mp
import numpy as np


def process_frame(cap, face_mesh):
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)

        both_eye_status = "Closed"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye_landmarks = get_eye_landmarks(face_landmarks.landmark, LEFT_EYE_INDICES)
                right_eye_landmarks = get_eye_landmarks(face_landmarks.landmark, RIGHT_EYE_INDICES)

                left_ear = calculate_ear(left_eye_landmarks)
                right_ear = calculate_ear(right_eye_landmarks)

                # Check if both eyes are open and draw landmarks
                if right_ear >= EAR_THRESHOLD and left_ear >= EAR_THRESHOLD:
                    both_eye_status = "Open"
                    print("Both eyes are open")
                    for idx in LEFT_EYE_INDICES:
                        x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                        y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                        # cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                    for idx in RIGHT_EYE_INDICES:
                        x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                        y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                        # cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                    # Proceed to the second part: detecting iris position
                    mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in face_landmarks.landmark])
                    (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                    (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                    center_left = np.array([l_cx, l_cy], dtype=np.int32)
                    center_right = np.array([r_cx, r_cy], dtype=np.int32)
                    
                    cv2.circle(frame, center_left, int(l_radius), (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(frame, center_right, int(r_radius), (0, 255, 255), 1, cv2.LINE_AA)
                    
                    cv2.circle(frame, mesh_points[R_H_RIGHT][0], 2, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(frame, mesh_points[R_H_LEFT][0], 2, (0, 255, 255), 1, cv2.LINE_AA)
                    
                    cv2.circle(frame, mesh_points[L_H_RIGHT][0], 2, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(frame, mesh_points[L_H_LEFT][0], 2, (0, 255, 255), 1, cv2.LINE_AA)
                    iris_pos, ratio = iris_position(center_right, mesh_points[R_H_RIGHT][0], mesh_points[R_H_LEFT][0])
                    
                    cv2.putText(frame, f"Iris Position: {iris_pos}", (50, 80), cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                else:
                    print("Eyes Closed")

        # Flip the frame horizontally for a mirror-like effect
        # frame = cv2.flip(frame, 1)

        # Add text after flipping
        cv2.putText(frame, f"Both Eyes: {both_eye_status}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if both_eye_status == "Open" else (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('MediaPipe Iris Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        process_frame(cap, face_mesh)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
