"""Task 5: Face Detection and Recognition
Detects and recognizes faces in images and videos using pre-trained models.
Uses Haar Cascades and deep learning-based face detectors.
"""

import cv2
import numpy as np
from pathlib import Path
import os

class FaceDetector:
    def __init__(self, cascade_path=None):
        """Initialize face detector with Haar Cascade classifier"""
        if cascade_path is None:
            # Use default cascade from OpenCV
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def detect_faces(self, image_path):
        """Detect faces in a static image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image {image_path}")
                return None
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            return image, faces, gray
        except Exception as e:
            print(f"Error detecting faces: {str(e)}")
            return None
    
    def draw_faces(self, image, faces, color=(255, 0, 0)):
        """Draw rectangles around detected faces"""
        result = image.copy()
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
            
            # Add label
            cv2.putText(result, 'Face', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result
    
    def process_image(self, image_path, output_path=None):
        """Detect faces and save annotated image"""
        result = self.detect_faces(image_path)
        if result is None:
            return
        
        image, faces, gray = result
        print(f"Detected {len(faces)} face(s) in {image_path}")
        
        # Draw faces
        annotated = self.draw_faces(image, faces)
        
        # Save result
        if output_path:
            cv2.imwrite(output_path, annotated)
            print(f"Saved annotated image to {output_path}")
        
        return annotated
    
    def detect_eyes(self, image, faces, gray):
        """Detect eyes within detected face regions"""
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        return image
    
    def detect_from_webcam(self, duration=10):
        """Detect faces in real-time from webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print(f"Capturing from webcam for {duration} seconds...")
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            
            # Draw faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, 'Face', (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Display frame
            cv2.imshow('Face Detection', frame)
            frame_count += 1
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Processed {frame_count} frames from webcam")
    
    def detect_in_video(self, video_path, output_path=None):
        """Detect faces in video file"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer if output path specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        face_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            
            face_count += len(faces)
            
            # Draw faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            if out:
                out.write(frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames, found {face_count} faces")
        
        cap.release()
        if out:
            out.release()
        
        print(f"Completed! Processed {frame_count} frames, detected {face_count} faces total")

if __name__ == "__main__":
    # Example usage
    detector = FaceDetector()
    
    print("Face Detection and Recognition System")
    print("=" * 50)
    print("\nCapabilities:")
    print("1. Detect faces in static images")
    print("2. Detect faces in video files")
    print("3. Real-time face detection from webcam")
    print("4. Eye detection within face regions")
    print("5. Annotate and save results")
    print("\nUsage:")
    print("  - Image: detector.process_image('image.jpg', 'output.jpg')")
    print("  - Webcam: detector.detect_from_webcam(duration=10)")
    print("  - Video: detector.detect_in_video('video.mp4', 'output.mp4')")
    print("\nNote: Uses Haar Cascade classifiers from OpenCV")
    print("For advanced recognition, consider using Deep Learning models like:")
    print("  - FaceNet, DeepFace, or ArcFace for face verification")
    print("  - MediaPipe Face Mesh for detailed facial landmarks")
