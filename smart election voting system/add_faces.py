import cv2
import pickle
import numpy as np
import os
import re

# Create directory if it doesn't exist
if not os.path.exists('data/'):
    os.makedirs('data/')

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

faces_data = []
framesTotal = 51  # Max frames
capture_frame = False

# Validate Voter ID input (10-character uppercase alphanumeric)
while True:
    voter_id = input("Enter your Voter ID Number (10 uppercase letters/numbers): ").strip()
    if re.match(r'^[A-Z0-9]{10}$', voter_id):
        break
    print("Invalid input! Voter ID must be exactly 10 characters (A-Z, 0-9).")

# Mouse event handler to capture frames
def on_mouse_click(event, x, y, flags, param):
    global capture_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        capture_frame = True

cv2.namedWindow("frame")
cv2.setMouseCallback("frame", on_mouse_click)

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
        cv2.putText(frame, voter_id, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2)

        if capture_frame and len(faces_data) < framesTotal:
            crop_img = frame[y:y + h, x:x + w]
            resized_img = cv2.resize(crop_img, (50, 50))
            faces_data.append(resized_img)
            capture_frame = False  # Reset flag

    # Show count of captured images
    cv2.putText(frame, f"Frames: {len(faces_data)} / {framesTotal}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
    cv2.imshow('frame', frame)

    # Exit conditions & capture on Enter key
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) >= framesTotal:
        break
    elif k == 13:  # Enter key pressed
        capture_frame = True

video.release()
cv2.destroyAllWindows()

# Convert to NumPy array and reshape (use actual captured frames count)
if len(faces_data) > 0:
    faces_data = np.asarray(faces_data)
    faces_data = faces_data.reshape((len(faces_data), -1))

    # Save voter ID along with captured face data
    if 'names.pkl' not in os.listdir('data/'):
        names = [voter_id] * len(faces_data)
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)
    else:
        with open('data/names.pkl', 'rb') as f:
            names = pickle.load(f)
        names += [voter_id] * len(faces_data)
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(names, f)

    # Save face data
    if 'faces_data.pkl' not in os.listdir('data/'):
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces_data, f)
    else:
        with open('data/faces_data.pkl', 'rb') as f:
            faces = pickle.load(f)
        faces = np.append(faces, faces_data, axis=0)
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces, f)

    print(f"Face data saved successfully ({len(faces_data)} frames).")
else:
    print("No frames were captured.")
