import cv2
import os

# Ensure the folder exists
SAVE_DIR = "known_faces"
os.makedirs(SAVE_DIR, exist_ok=True)

# Start the webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Face Registration")

print("Press 's' to save the face, or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    cv2.imshow("Face Registration", frame)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('s'):
        name = input("Enter the name for this face (no spaces): ").strip().lower()
        filename = os.path.join(SAVE_DIR, f"{name}.jpg")
        cv2.imwrite(filename, frame)
        print(f"✅ Face saved as {filename}")
        break
    elif key == ord('q'):
        print("❌ Registration cancelled.")
        break

cap.release()
cv2.destroyAllWindows()
