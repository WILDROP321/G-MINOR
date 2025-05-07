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
from Grock_api import chat
from config import EMOTION, PERSON
from collections import Counter


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


def get_finger_status(hand_landmarks, handedness_label):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb: direction differs by hand
    if handedness_label == "Right":
        fingers.append(1 if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x else 0)
    else:
        fingers.append(1 if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else 0)

    for tip in tips[1:]:
        fingers.append(1 if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y else 0)

    return fingers


def detect_gesture(fingers):
    known_gestures = {
        (1, 0, 0, 0, 1): "🤙  (Shaka sign)",
        (1, 1, 0, 0, 1): "🤟 (Rock)",
        (0, 1, 1, 0, 0): "✌️(Peace)",
        (0, 0, 1, 1, 1): "👌 (Awesome)",
        (1, 1, 1, 1, 1): "✋(Hand)",
        (0, 0, 1, 0, 0): "🖕 (Middle Finger)",
        (1, 0, 0, 0, 0): "👍"
    }

    def hamming(a, b):
        return sum(x != y for x, y in zip(a, b))

    best_match = min(known_gestures.keys(), key=lambda k: hamming(k, tuple(fingers)))
    if hamming(best_match, tuple(fingers)) <= 1:
        return known_gestures[best_match]

    return f"Unknown Gesture: {fingers}"

def run_gesture_detection(gui_callback=None, video_callback=None, stop_flag=lambda: False, detect_gestures=True, finger_count_signal=None, camera_index=0):
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
                continue

            small_frame = cv2.resize(frame, (640, 580))
            img_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if cooldown > 0:
                cooldown -= 1

            if results.multi_hand_landmarks:
                for idx, handLms in enumerate(results.multi_hand_landmarks):
                    handedness_label = results.multi_handedness[idx].classification[0].label
                    mp_draw.draw_landmarks(small_frame, handLms, mp_hands.HAND_CONNECTIONS)
                    fingers = get_finger_status(handLms, handedness_label)
                    gesture = detect_gesture(fingers)
                    gesture_queue.append(gesture)

                    # Debug overlay: show fingers and gesture on screen
                    debug_text = f"{fingers} → {gesture}"
                    cv2.putText(small_frame, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    # Apply smoothing: trigger most common gesture if queue is full
                    if len(gesture_queue) == gesture_queue.maxlen:
                        most_common_gesture = Counter(gesture_queue).most_common(1)[0][0]
                        if most_common_gesture != last_displayed and cooldown == 0:
                            last_displayed = most_common_gesture
                            cooldown = COOLDOWN_TIME
                            print(f"✅ Smoothed Gesture: {most_common_gesture}")
                            # You can call speak() or send signal here if needed


            if video_callback:
                h, w, ch = small_frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(small_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                video_callback.emit(qt_image)

        cap.release()
        cv2.destroyAllWindows()


def process_single_frame(frame, gui_callback=None):
    known_encodings, known_names = load_known_faces()
    small_frame = frame.copy()
    img_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for idx, handLms in enumerate(results.multi_hand_landmarks):
            handedness_label = results.multi_handedness[idx].classification[0].label
            fingers = get_finger_status(handLms, handedness_label)
            gesture = detect_gesture(fingers)
            person = "Unknown"
            message = None

            if gesture:
                person = identify_person(frame, known_encodings, known_names)
                msg = f"Detected: {gesture} by {person}"
                print("MESSAGE :", msg)

                system_prompt = PERSON.get(person.lower())
                query_text = f"{gesture}" if person == 'default' else f"{gesture} by {person}"

                try:
                    print(f"Query: {query_text}")
                    message = chat(query_text, system_prompt)
                    print(f"Chatbot response: {message}")
                except Exception as e:
                    message = f"{gesture} detected, but chatbot failed."

                display_text = msg  # Set the base detection message
                if message:
                    display_text += f"\nBot: {message}"  # Append bot's response

                if gui_callback:
                    combined_data = json.dumps({
                        "gesture": gesture,
                        "display_text": display_text
                    })
                    gui_callback.emit(combined_data)

                speak(message or gesture)
            else:
                if gui_callback:
                    gui_callback.emit("No valid gesture detected.")
    else:
        if gui_callback:
            gui_callback.emit("No hand detected.")
