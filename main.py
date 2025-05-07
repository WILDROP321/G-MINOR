import sys
import threading
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout,
    QLabel, QLineEdit
)
from PyQt5.QtCore import Qt, QTimer
from Grock_api import chat  # Your chatbot API function
from gtts import gTTS
from playsound import playsound
import tempfile
from Gesture_rec import run_gesture_detection, identify_person  # Your gesture detection function
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QTabWidget
from config import EMOTION  # Your config file
from PyQt5.QtWidgets import QFrame
import numpy as np
import json

# Text-to-speech setup 
def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=True, suffix='.mp3') as fp:
            tts.save(fp.name)
            playsound(fp.name)
    except Exception as e:
        print("TTS failed:", e)

class AIChatApp(QWidget):
    gesture_detected = pyqtSignal(str)
    video_updated = pyqtSignal(QImage)
    clear_video = pyqtSignal()
    def __init__(self):
            super().__init__()
            self.setWindowTitle("EchoFrame: Emotionally Adaptive Bot")
            QLabel("A metaphoric bridge between machine behavior and human emotion")
            self.setGeometry(100, 100, 400, 400)

            self.camera_stop_event = threading.Event()
            self.clear_video.connect(self.clear_video_frame)

            self.camera_thread = None
            self.camera_running = False

            self.camera_enabled = True

            self.input_box = QLineEdit(self)
            self.input_box.setPlaceholderText("Ask something...")

            self.submit_button = QPushButton("Submit", self)
            self.submit_button.clicked.connect(self.AI_chat)

            self.response_label = QLabel("", self)
            self.response_label.setWordWrap(True)
            self.response_label.setAlignment(Qt.AlignTop)

            self.gesture_label = QLabel("")
            self.gesture_label.setAlignment(Qt.AlignTop)

            self.video_label = QLabel("Camera feed will show here.")
            self.video_label.setFixedSize(640, 340)

            self.camera_checkbox = QCheckBox("Enable Camera Gesture Detection", self)
            self.camera_checkbox.setChecked(True)
            self.camera_checkbox.stateChanged.connect(self.toggle_camera)

            self.snapshot_button = QPushButton("Take Snapshot for Gesture", self)
            self.snapshot_button.clicked.connect(self.capture_gesture_from_frame)

            self.emotion_tabs = QTabWidget(self)
            self.emotion_tabs.setTabPosition(QTabWidget.North)

            # Create one tab per emotion
            for emotion in EMOTION.keys():
                tab = QWidget()
                self.emotion_tabs.addTab(tab, emotion.title())

            layout = QVBoxLayout()


            # SECTION 1: Emotion Selector
            title_emotion = QLabel("<b>Select Bot Emotion:</b>")
            layout.addWidget(title_emotion)
            layout.addWidget(self.emotion_tabs)

            # Divider
            line1 = QFrame()
            line1.setFrameShape(QFrame.HLine)
            line1.setFrameShadow(QFrame.Sunken)
            layout.addWidget(line1)

            # SECTION 2: User Input
            title_input = QLabel("<b>Ask the Bot:</b>")
            layout.addWidget(title_input)
            layout.addWidget(self.input_box)
            layout.addWidget(self.submit_button)

            # Divider
            line2 = QFrame()
            line2.setFrameShape(QFrame.HLine)
            line2.setFrameShadow(QFrame.Sunken)
            layout.addWidget(line2)

            # SECTION 3: Bot Response
            title_response = QLabel("<b>Bot Response:</b>")
            layout.addWidget(title_response)
            layout.addWidget(self.response_label)

            # Divider
            line3 = QFrame()
            line3.setFrameShape(QFrame.HLine)
            line3.setFrameShadow(QFrame.Sunken)
            layout.addWidget(line3)

            # SECTION 4: Gesture Detection
            title_gesture = QLabel("<b>Gesture Detection:</b>")
            layout.addWidget(title_gesture)
            layout.addWidget(self.gesture_label)

            # Divider
            line4 = QFrame()
            line4.setFrameShape(QFrame.HLine)
            line4.setFrameShadow(QFrame.Sunken)
            layout.addWidget(line4)

            # SECTION 5: Video Feed
            title_video = QLabel("<b>Live Camera Feed:</b>")
            layout.addWidget(title_video)
            layout.addWidget(self.video_label)
            layout.addWidget(self.camera_checkbox)
            
            layout.addWidget(self.snapshot_button)

            self.setLayout(layout)
            self.latest_frame = None

            self.gesture_detected.connect(self.update_gesture_label)
            self.video_updated.connect(self.update_video_frame)
            QTimer.singleShot(500, lambda: self.input_box.setFocus())
    
    def clear_video_frame(self):
        blank = QPixmap(self.video_label.width(), self.video_label.height())
        blank.fill(Qt.black)
        self.video_label.setPixmap(blank)

    def toggle_camera(self, state):
        if state == Qt.Checked and not self.camera_running:
            self.camera_running = True
            self.camera_enabled = True
            self.camera_stop_event.clear()
            self.camera_thread = threading.Thread(
                target=run_gesture_detection,
                args=(
                    self.gesture_detected,
                    self.video_updated,
                    self.camera_stop_event.is_set
                ),
                daemon=True
            )
            self.camera_thread.start()

        elif state != Qt.Checked and self.camera_running:
            self.camera_running = False
            self.camera_enabled = False
            self.camera_stop_event.set()

            # Replace label with a black frame
            blank = QPixmap(self.video_label.width(), self.video_label.height())
            blank.fill(Qt.black)
            self.video_label.setPixmap(blank)

    def update_gesture_label(self, data_str):
        try:
            data = json.loads(data_str)
            gesture = data.get("gesture", "Unknown gesture")
            display_text = data.get("display_text", "")
            self.gesture_label.setText(f"{gesture}\n\n{display_text}")
        except json.JSONDecodeError:
            # If it fails to decode, show raw string
            self.gesture_label.setText(data_str)

    
    def capture_gesture_from_frame(self):
        if self.latest_frame is not None:
            from Gesture_rec import process_single_frame
            processed_frame = process_single_frame(self.latest_frame, gui_callback=self.gesture_detected)


    def update_video_frame(self, qimage):
        if self.camera_enabled:
            self.video_label.setPixmap(QPixmap.fromImage(qimage))
            # Save frame for snapshot
            image = qimage.convertToFormat(QImage.Format_RGB888)
            width, height = image.width(), image.height()
            ptr = image.bits()
            ptr.setsize(image.byteCount())
            self.latest_frame = np.array(ptr).reshape(height, width, 3)

    def AI_chat(self):
        query = self.input_box.text().strip()
        if query:
            try:
                current_emotion = self.emotion_tabs.tabText(self.emotion_tabs.currentIndex()).upper()
                emotion_prompt = EMOTION.get(current_emotion, EMOTION["WITTY"])  # Fallback
                response = chat(str(query), emotion_prompt)  # Make sure chat() takes (query, prompt)
                self.response_label.setText(response)
                speak(response)
            except Exception as e:
                self.response_label.setText(f"Error: {e}")
        else:
            self.response_label.setText("Please enter something first."
            )
    
        self.input_box.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AIChatApp()
    win.show()
    win.raise_()
    win.activateWindow()

    sys.exit(app.exec_())