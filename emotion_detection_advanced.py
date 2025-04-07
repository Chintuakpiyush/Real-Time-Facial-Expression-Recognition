from deepface import DeepFace
import cv2
import pandas as pd
import time

# Initialize variables
emotions_history = []
frame_count = 0
start_time = time.time()

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process every 5th frame for better performance
    if frame_count % 5 == 0:
        try:
            # Analyze with multiple models (uncomment if needed)
            # result = DeepFace.analyze(frame, actions=['emotion'], model_name='Facenet')
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            
            # Get all emotion scores
            emotions = result[0]['emotion']
            emotions_history.append(emotions)
            
            # Display dominant emotion
            dominant_emotion = result[0]['dominant_emotion']
            cv2.putText(frame, f"Dominant: {dominant_emotion}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display all emotion scores
            for idx, (emotion, score) in enumerate(emotions.items()):
                cv2.putText(frame, f"{emotion}: {score:.1f}%", (10, 70 + idx * 25), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                
        except Exception as e:
            print(f"Error: {e}")
    
    # Display FPS
    fps = frame_count / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 470), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    frame_count += 1
    cv2.imshow("Advanced Emotion Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save emotion history to CSV
if emotions_history:
    df = pd.DataFrame(emotions_history)
    df.to_csv('emotion_log.csv', index=False)
    print("Emotion data saved to emotion_log.csv")

cap.release()
cv2.destroyAllWindows()
