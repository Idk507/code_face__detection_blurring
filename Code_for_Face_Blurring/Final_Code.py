import cv2
import mediapipe as mp

import numpy as np

from facial_landmarks import FaceLandmarks

fl = FaceLandmarks()
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("PicsArt_10-13-09.58.45.mp4")

while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5)

    frame_copy = frame.copy()

    frame_copy = cv2.blur(frame_copy, (27, 27))

    height, width, _ = frame.shape

    landmarks = fl.get_facial_landmarks(frame)

    print(len(landmarks))

    pt = landmarks[0]

    convexhull = cv2.convexHull(landmarks)

    mask = np.zeros((height, width), np.uint8)
    #cv2.polylines(frame, [convexhull], True, (0, 255, 0), 3)



    cv2.fillConvexPoly(mask, convexhull, 255)

    face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask = mask)
    # Blurring the Face

    blurred_face = cv2.GaussianBlur(face_extracted, (27, 27), 0)


    # Extract the Back Ground

    background_mask = cv2.bitwise_not(mask)


    background = cv2.bitwise_and(frame, frame, mask = background_mask)


    result = cv2.add(background, face_extracted)

    cv2.imshow("Final Result", result)

    #cv2.imshow("Background", background)


    #cv2.imshow("Mask Inverse", background_mask)
    #cv2.imshow("Blurred Face", blurred_face)

    cv2.imshow("Frame", frame)

    #cv2.imshow("Face Extracted", face_extracted)

    #cv2.imshow("Mask", mask)

    key = cv2.waitKey(30) & 0xFF
    if key == 2:
        break

cap.release()





