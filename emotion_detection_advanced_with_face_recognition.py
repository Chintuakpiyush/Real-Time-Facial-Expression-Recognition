from deepface import DeepFace
import cv2
import pandas as pd
import time
import os

# Initialize variables
emotions_history = []
frame_count = 0
start_time = time.time()
last_valid_emotion = {"emotion": {"angry": 0, "disgust": 0, "fear": 0, "happy": 0, 
                                 "sad": 0, "surprise": 0, "neutral": 0},
                     "dominant_emotion": "neutral"}

# Performance settings
PROCESS_EVERY = 5  # Process every Nth frame
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# Create face database directory
if not os.path.exists('face_database'):
    os.makedirs('face_database')
    print("Created 'face_database' folder. Add reference images of people here.")

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

# Warm up camera
for _ in range(5):
    ret, frame = cap.read()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    display_frame = frame.copy()
    
    # Process analysis in selected frames
    if frame_count % PROCESS_EVERY == 0:
        try:
            # Analyze emotions (single face only for better performance)
            emotion_result = DeepFace.analyze(frame, actions=['emotion'], 
                                           enforce_detection=False, 
                                           detector_backend='opencv')
            
            if emotion_result:
                last_valid_emotion = emotion_result[0]
                emotions_history.append(last_valid_emotion['emotion'])
            
            # Face recognition (skip if database is empty)
            if os.path.exists('face_database') and len(os.listdir('face_database')) > 0:
                recognition = DeepFace.find(frame, db_path="face_database", 
                                          enforce_detection=False,
                                          detector_backend='opencv')
                if recognition and len(recognition[0]) > 0:
                    identity = os.path.splitext(os.path.basename(recognition[0]['identity'][0]))[0]
                    cv2.putText(display_frame, f"Person: {identity}", (400, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
        except Exception as e:
            print(f"Analysis error: {e}")
    
    # Always display the last valid emotion results
    emotions = last_valid_emotion['emotion']
    dominant_emotion = last_valid_emotion['dominant_emotion']
    
    # Display dominant emotion
    cv2.putText(display_frame, f"Dominant: {dominant_emotion}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display all emotion scores
    for idx, (emotion, score) in enumerate(emotions.items()):
        cv2.putText(display_frame, f"{emotion}: {score:.1f}%", (10, 70 + idx * 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    # Display FPS
    fps = frame_count / (time.time() - start_time)
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, DISPLAY_HEIGHT - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    frame_count += 1
    cv2.imshow("Emotion + Face Recognition", display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save emotion history to CSV
if emotions_history:
    df = pd.DataFrame(emotions_history)
    df.to_csv('emotion_log.csv', index=False)
    print(f"Saved {len(df)} emotion records to emotion_log.csv")

cap.release()
cv2.destroyAllWindows()
