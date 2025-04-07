from deepface import DeepFace
import cv2
import pandas as pd
import time
import os
import numpy as np

# Configuration
PROCESS_EVERY = 5  # Process every Nth frame
DISPLAY_WIDTH = 1920
DISPLAY_HEIGHT = 1080
MIN_FACE_CONFIDENCE = 0.7  # Minimum confidence to consider a face valid
FONT_SCALE = 0.6
FONT_THICKNESS = 1

# Initialize variables
emotions_history = []
frame_count = 0
start_time = time.time()
last_valid_results = []

# Create face database
if not os.path.exists('face_database'):
    os.makedirs('face_database')
    print("Created 'face_database' folder. Add reference images named like 'John.jpg'")

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

# Performance optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_person_name(face_image):
    """Identify person from face database"""
    if not os.path.exists('face_database') or not os.listdir('face_database'):
        return "Unknown"
    
    try:
        recognition = DeepFace.find(face_image, 
                                 db_path="face_database",
                                 enforce_detection=False,
                                 detector_backend='opencv',
                                 silent=True)
        if recognition and len(recognition[0]) > 0:
            if recognition[0]['distance'][0] < 0.5:  # Confidence threshold
                return os.path.splitext(os.path.basename(recognition[0]['identity'][0]))[0]
    except:
        pass
    return "Unknown"

def get_face_area(face_obj):
    """Safely extract face coordinates from detection result"""
    try:
        area = face_obj['facial_area']
        if isinstance(area, dict):
            return area['x'], area['y'], area['w'], area['h']
        elif isinstance(area, (list, tuple)):
            return area[0], area[1], area[2], area[3]
    except:
        return None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    display_frame = frame.copy()
    current_results = []
    
    # Process analysis in selected frames
    if frame_count % PROCESS_EVERY == 0:
        try:
            # Detect all faces first for better performance
            face_objs = DeepFace.extract_faces(frame, 
                                             detector_backend='opencv',
                                             enforce_detection=False,
                                             align=False)
            
            for face_obj in face_objs:
                if face_obj['confidence'] < MIN_FACE_CONFIDENCE:
                    continue
                
                # Safely get face coordinates
                face_coords = get_face_area(face_obj)
                if not face_coords:
                    continue
                    
                x, y, w, h = face_coords
                face_img = frame[y:y+h, x:x+w]
                
                # Get emotion (fast analysis)
                try:
                    emotion_result = DeepFace.analyze(face_img, 
                                                   actions=['emotion'],
                                                   enforce_detection=False,
                                                   detector_backend='skip',
                                                   silent=True)
                    
                    if emotion_result:
                        # Get person name
                        person_name = get_person_name(face_img)
                        
                        current_results.append({
                            'box': (x, y, w, h),
                            'emotion': emotion_result[0]['emotion'],
                            'dominant_emotion': emotion_result[0]['dominant_emotion'],
                            'name': person_name
                        })
                except Exception as e:
                    print(f"Emotion analysis error: {e}")
                
            if current_results:
                last_valid_results = current_results
                emotions_history.extend([r['emotion'] for r in current_results])
                
        except Exception as e:
            print(f"Face detection error: {e}")
    
    # Display last valid results
    for i, result in enumerate(last_valid_results):
        x, y, w, h = result['box']
        
        # Draw face bounding box
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display name above head
        cv2.putText(display_frame, result['name'], (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 0), FONT_THICKNESS)
        
        # Display dominant emotion below face
        cv2.putText(display_frame, result['dominant_emotion'], (x, y+h+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 255, 255), FONT_THICKNESS)
        
        # Display person number
        cv2.putText(display_frame, f"Person {i+1}", (x, y-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 165, 255), FONT_THICKNESS)
    
    # Display summary info
    summary_text = f"People: {len(last_valid_results)} | FPS: {frame_count/(time.time()-start_time):.1f}"
    cv2.putText(display_frame, summary_text, (10, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    frame_count += 1
    cv2.imshow("Multi-Person Emotion Recognition", display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save data
if emotions_history:
    df = pd.DataFrame(emotions_history)
    df.to_csv('emotion_log.csv', index=False)
    print(f"Saved {len(df)} emotion records")

cap.release()
cv2.destroyAllWindows()
