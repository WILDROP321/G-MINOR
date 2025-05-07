import cv2
import mediapipe as mp
from gtts import gTTS
from playsound import playsound
import tempfile
import os
import json
import random
import face_recognition
import numpy as np
from collections import deque
from PyQt5.QtGui import QImage
from Grock_api import chat  # Your chatbot API function
from config import EMOTION, PERSON


def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=True, suffix='.mp3') as fp:
            tts.save(fp.name)
            playsound(fp.name)
    except Exception as e:
        print("TTS failed:", e)


def load_known_faces(folder="known_faces"):
    known_encodings = []
    known_names = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = face_recognition.load_image_file(os.path.join(folder, filename))
            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])
    return known_encodings, known_names


def identify_person(frame, known_encodings, known_names):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, faces)
    for encoding in encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
        if True in matches:
            index = matches.index(True)
            return known_names[index]
    return "default"


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


def get_finger_status(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    for tip in tips[1:]:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers


def detect_gesture(fingers):
    if fingers == [1, 0, 0, 0, 1]: return "🤙  (Shaka sign)"
    if fingers == [1, 1, 0, 0, 1]: return "🤟 (Rock)"
    if fingers == [0, 1, 1, 0, 0]: return "✌️(Peace)"
    if fingers == [0, 0, 1, 1, 1]: return "👌"
    if fingers == [1, 1, 1, 1, 1]: return "✋(hand)"
    if fingers == [0, 0, 1, 0, 0]: return "🖕 (Middle Finger)"
    if fingers == [1, 0, 0, 0, 0]: return "👍"

    return f"Unknown Gesture: {fingers}"


def run_gesture_detection(gui_callback=None, video_callback=None, stop_flag=lambda: False, detect_gestures=True, finger_count_signal=None, camera_index=1):
    if detect_gestures:
        gesture_queue = deque(maxlen=5)
        last_displayed = None
        cooldown = 0
        COOLDOWN_TIME = 5

        cap = cv2.VideoCapture(camera_index)
        known_encodings, known_names = load_known_faces()

        while not stop_flag():
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ Failed to grab frame. Skipping...")
                continue  # Skip this iteration safely

            if frame is None:
                if gui_callback:
                    gui_callback.emit("Snapshot failed — no frame.")
                return

            small_frame = cv2.resize(frame, (640, 580))


            img_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if cooldown > 0:
                cooldown -= 1

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(small_frame, handLms, mp_hands.HAND_CONNECTIONS)
                    fingers = get_finger_status(handLms)
                    gesture = detect_gesture(fingers)
                    gesture_queue.append(gesture)

            if video_callback:
                h, w, ch = small_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(small_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                video_callback.emit(qt_image)

        cap.release()
        cv2.destroyAllWindows()


def process_single_frame(frame, gui_callback=None):
    known_encodings, known_names = load_known_faces()
    small_frame = cv2.resize(frame, (640, 480))
    img_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            fingers = get_finger_status(handLms)
            gesture = detect_gesture(fingers)
            person = "Unknown"
            message = None

            if gesture:
                person = identify_person(frame, known_encodings, known_names).capitalize()
                msg = f"Detected: {gesture} by {person}"

                use_name = random.choice([True, False])
                system_prompt = PERSON.get(person.lower(), random.choice(list(EMOTION.values())))
                query_text = f"{gesture} by {person}" if use_name else gesture

                try:
                    message = chat(query_text, system_prompt)
                except Exception as e:
                    message = f"{gesture} detected, but chatbot failed."

                # Combine display
                display_text = msg
                if message:
                    display_text += f"\nBot: {message}"

                if gui_callback:
                    gui_callback.emit(display_text)
                speak(message or gesture)
            else:
                if gui_callback:
                    gui_callback.emit("No valid gesture detected.")
    else:
        if gui_callback:
            gui_callback.emit("No hand detected.")

