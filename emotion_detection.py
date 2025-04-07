from deepface import DeepFace
import cv2

cap = cv2.VideoCapture(0)  # Use webcam

while True:
    ret, frame = cap.read()
    
    try:
        # Detect emotion
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        
        # Display emotion
        cv2.putText(frame, f"Emotion: {emotion}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except:
        pass
    
    cv2.imshow("Facial Expression Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
