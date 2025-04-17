from deepface import DeepFace
import cv2
import pandas as pd
import time
import os
import numpy as np

# Configuration
PROCESS_EVERY = 30                # Process every Nth frame
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480
MIN_FACE_CONFIDENCE = 0.7        # Minimum face detection confidence
RECOGNITION_THRESHOLD = 0.5      # DeepFace distance threshold
FONT_SCALE = 0.6
FONT_THICKNESS = 1

# Initialize
emotions_history = []
frame_count = 0
start_time = time.time()
last_valid_results = []

# Ensure face database exists
if not os.path.exists('face_database'):
    os.makedirs('face_database')
    print("Created 'face_database'. Add reference images named like 'John.jpg'")

# Print the absolute path for clarity
print("Face database path:", os.path.abspath("face_database"))

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_person_name(face_image):
    """Identify person from face_database by temporarily saving the crop."""
    if not os.listdir('face_database'):
        return "Unknown"
    
    temp_img_path = "temp_face.jpg"
    try:
        # Save the cropped face
        cv2.imwrite(temp_img_path, face_image)

        # Run DeepFace.find on the saved file
        recognition = DeepFace.find(
            img_path=temp_img_path,
            db_path="face_database",
            enforce_detection=False,
            detector_backend='opencv',
            silent=True
        )

        # If any match is found and within threshold, return the basename
        if recognition and len(recognition[0]) > 0:
            best_match = recognition[0].iloc[0]
            distance = best_match['distance']
            if distance < RECOGNITION_THRESHOLD:
                identity_path = best_match['identity']
                return os.path.splitext(os.path.basename(identity_path))[0]

    except Exception as e:
        print(f"Recognition error: {e}")
    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)

    return "Unknown"

def get_face_area(face_obj):
    """Extract face bounding box coordinates from DeepFace.extract_faces output."""
    try:
        area = face_obj['facial_area']
        if isinstance(area, dict):
            return area['x'], area['y'], area['w'], area['h']
        elif isinstance(area, (list, tuple)):
            return area[0], area[1], area[2], area[3]
    except:
        pass
    return None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    current_results = []

    # Only analyze every Nth frame for performance
    if frame_count % PROCESS_EVERY == 0:
        try:
            # Detect all faces
            face_objs = DeepFace.extract_faces(
                frame,
                detector_backend='opencv',
                enforce_detection=False,
                align=False
            )

            for face_obj in face_objs:
                if face_obj['confidence'] < MIN_FACE_CONFIDENCE:
                    continue

                coords = get_face_area(face_obj)
                if not coords:
                    continue

                x, y, w, h = coords
                face_img = frame[y:y+h, x:x+w]

                # Get emotion
                try:
                    emotion_res = DeepFace.analyze(
                        face_img,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='skip',
                        silent=True
                    )
                    if emotion_res:
                        name = get_person_name(face_img)
                        current_results.append({
                            'box': (x, y, w, h),
                            'dominant_emotion': emotion_res[0]['dominant_emotion'],
                            'name': name
                        })
                except Exception as e:
                    print(f"Emotion analysis error: {e}")

            if current_results:
                last_valid_results = current_results
                # log full emotion dictionaries if needed; here we log only dominant emotions
                emotions_history.extend([r['dominant_emotion'] for r in current_results])

        except Exception as e:
            print(f"Face detection error: {e}")

    # Draw results from the last valid analysis
    for i, res in enumerate(last_valid_results):
        x, y, w, h = res['box']

        # Bounding box
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Labels: Person ID, Name, Emotion
        cv2.putText(display_frame, f"Person {i+1}", (x, y-40),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)
        cv2.putText(display_frame, f"Name: {res['name']}", (x, y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 255), FONT_THICKNESS)
        cv2.putText(display_frame, f"Emotion: {res['dominant_emotion']}", (x, y+h+20),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 165, 255), FONT_THICKNESS)

    # Summary overlay
    fps = frame_count / (time.time() - start_time) if frame_count > 0 else 0.0
    summary = f"People: {len(last_valid_results)} | FPS: {fps:.1f}"
    cv2.putText(display_frame, summary, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    frame_count += 1
    cv2.imshow("Real-Time Face Recognition + Emotion Detection", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save emotion log
if emotions_history:
    df = pd.DataFrame(emotions_history, columns=['dominant_emotion'])
    df.to_csv('emotion_log.csv', index=False)
    print(f"Saved {len(df)} emotion records to emotion_log.csv")

cap.release()
cv2.destroyAllWindows()
print("Webcam closed. Exiting...")
print("Total frames processed:", frame_count)
