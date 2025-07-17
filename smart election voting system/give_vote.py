from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

def check_voter_id(voter_id):
    """Check if the voter has already voted."""
    try:
        with open("Votes.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row and row[0] == voter_id:
                    return True
    except FileNotFoundError:
        return False
    return False

# Step 1: Ask for Voter ID
voter_id = input("Enter your Voter ID: ").strip()

if check_voter_id(voter_id):
    speak("Access Denied! You have already voted.")
    print("❌ Access Denied: This Voter ID has already voted. Exiting...")
    exit()

speak("Voter ID verified. Please look at the camera for face verification.")

# Initialize Camera and Face Recognition
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not os.path.exists('data/'):
    os.makedirs('data/')

with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)

with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

imgBackground = cv2.imread("background.png")

COL_NAMES = ['VOTER_ID', 'VOTE', 'DATE', 'TIME']

# Step 2: Face Verification (Only When Enter is Pressed)
verified = False
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.putText(frame, "Press Enter to Verify", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2)

    imgBackground[370:370 + 480, 225:225 + 640] = frame
    cv2.imshow('frame', imgBackground)

    k = cv2.waitKey(1)

    # Press Enter to capture and verify the face
    if k == 13:  # ASCII for Enter Key
        for (x, y, w, h) in faces:
            crop_img = frame[y:y+h, x:x+w]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)

            if output[0] == voter_id:
                verified = True
                speak("Face verified successfully. You can now vote.")
                break

    if verified:
        break
    elif k == ord('q'):  # Press 'q' to exit
        video.release()
        cv2.destroyAllWindows()
        exit()

if not verified:
    speak("Access Denied! Face verification failed.")
    print("❌ Access Denied: Face verification failed. Exiting...")
    video.release()
    cv2.destroyAllWindows()
    exit()

# Step 3: Voting Process
vote_dict = {
    ord('1'): "BJP",
    ord('2'): "CONGRESS",
    ord('3'): "AAP",
    ord('4'): "NOTA"
}

speak("Press 1 for BJP, 2 for Congress, 3 for AAP, 4 for NOTA.")

while True:
    k = cv2.waitKey(1)
    if k in vote_dict:
        vote = vote_dict[k]
        speak(f"Your vote has been recorded.")
        time.sleep(2)

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")

        exist = os.path.isfile("Votes.csv")
        with open("Votes.csv", "a") as csvfile:
            writer = csv.writer(csvfile)
            if not exist:
                writer.writerow(COL_NAMES)
            writer.writerow([voter_id, vote, date, timestamp])

        speak("Thank you for participating in the elections.")
        print(f"✅ Vote Recorded: {voter_id} voted on {date} at {timestamp}")
        break

video.release()
cv2.destroyAllWindows()
