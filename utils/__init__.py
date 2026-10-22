"""
Utility functions for Facial Expression Classification
Includes data preprocessing, visualization, and helper functions
"""
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
from datetime import datetime

class EmotionConfig:
    """Configuration class for emotion classification"""
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    NUM_CLASSES = 7
    IMG_SIZE = 48
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    @staticmethod
    def get_emotion_label(index):
        return EmotionConfig.EMOTIONS[index] if 0 <= index < len(EmotionConfig.EMOTIONS) else "Unknown"
    
    @staticmethod
    def get_emotion_index(label):
        try:
            return EmotionConfig.EMOTIONS.index(label.capitalize())
        except ValueError:
            return -1

def load_haarcascade():
    """Load Haar Cascade classifier for face detection"""
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            raise Exception("Failed to load Haar Cascade classifier")
        return face_cascade
    except Exception as e:
        raise Exception(f"Error loading face cascade: {str(e)}")

def preprocess_image(image, img_size=48):
    """
    Preprocess image for model prediction
    - Convert to grayscale
    - Resize to img_size
    - Normalize
    """
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        resized = cv2.resize(gray, (img_size, img_size))
        normalized = resized / 255.0
        reshaped = normalized.reshape(1, img_size, img_size, 1)
        return reshaped, gray
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def detect_faces(image, face_cascade, scale_factor=1.3, min_neighbors=5):
    """Detect faces in image using Haar Cascade"""
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(30, 30)
        )
        return faces, gray
    except Exception as e:
        raise Exception(f"Error detecting faces: {str(e)}")

def plot_training_history(history, save_path=None):
    """Plot training and validation accuracy/loss"""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss plot
        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Validation Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error plotting history: {str(e)}")

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """Plot confusion matrix"""
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return cm
    except Exception as e:
        print(f"Error plotting confusion matrix: {str(e)}")
        return None

def save_training_report(history, y_true, y_pred, classes, save_dir='models'):
    """Save training report with metrics"""
    try:
        os.makedirs(save_dir, exist_ok=True)
        report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(save_dir, f'training_report_{timestamp}.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"Training report saved to {report_path}")
        return report
    except Exception as e:
        print(f"Error saving training report: {str(e)}")
        return None
