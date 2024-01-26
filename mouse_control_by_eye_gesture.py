import cv2
import mediapipe as mp
import pyautogui as pg

cam = cv2.VideoCapture(0)
face_mesh_landmarks = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
while True:
    _,image = cam.read()
    window_h, window_w, _ = image.shape
    rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    processed_image = face_mesh_landmarks.process(rgb_image)
    all_face_landmark_points = processed_image.multi_face_landmarks
    if all_face_landmark_points:
        one_face_landmark_points =  all_face_landmark_points[0].landmark
        for landmark_point in one_face_landmark_points:
            print(landmark_point.x,landmark_point.y)
    cv2.imshow("Eye Controlled Mouse", image)
    key = cv2.waitKey(100)
    if key == 25:
        break
cam.release()
cv2.destroyAllWindows()
