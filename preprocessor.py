import cv2
import numpy as np
from mtcnn import MTCNN
import os
import re
import math

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def get_next_filename(base="image", ext=".npy"):
    files = os.listdir(".")
    numbers = [int(re.findall(rf"{base}(\d+){ext}", f)[0]) 
               for f in files if re.match(rf"{base}\d+{ext}", f)]
    next_num = max(numbers) + 1 if numbers else 1
    return next_num

def align_face(image, left_eye, right_eye):
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))

    eyes_center = ((left_eye[0] + right_eye[0]) // 2,
                   (left_eye[1] + right_eye[1]) // 2)
    
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1)
    aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                             flags=cv2.INTER_CUBIC)
    return aligned

def preprocess_and_save_all_faces(frame):
    detector = MTCNN()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_clahe = apply_clahe(frame_rgb)
    results = detector.detect_faces(frame_clahe)
    if not results:
        print("No faces detected.")
        return

    next_index = get_next_filename()

    for i, face_data in enumerate(results):
        x, y, w, h = face_data['box']
        x, y = abs(x), abs(y)

        keypoints = face_data['keypoints']
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']

        aligned_image = align_face(frame_rgb, left_eye, right_eye)
        face = aligned_image[y:y+h, x:x+w]

        try:
            face = cv2.resize(face, (160, 160), interpolation=cv2.INTER_AREA)
        except:
            print(f"Skipping face {i+1} due to bad crop.")
            continue

        face = face.astype('float32')
        face = (face - 127.5) / 128.0

        filename = f"image{next_index + i}.npy"
        np.save(filename, face)
        print(f"Saved aligned face {i+1} as {filename}")

def capture_from_webcam():
    cap = cv2.VideoCapture(0)
    print("Press SPACE to capture, ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera not accessible.")
            break

        cv2.imshow("Webcam - Press SPACE to Capture", frame)
        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            preprocess_and_save_all_faces(frame)
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_from_webcam()
