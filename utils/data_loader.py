"""
Data loading and preprocessing utilities
"""
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2

class DataLoader:
    def __init__(self, data_path='data', img_size=48):
        self.data_path = data_path
        self.img_size = img_size
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
    def load_from_directory(self):
        """
        Load images from directory structure:
        data/train/class_name/*.png or data/test/class_name/*.png
        """
        try:
            X_train, y_train = [], []
            X_test, y_test = [], []
            
            train_path = os.path.join(self.data_path, 'train')
            test_path = os.path.join(self.data_path, 'test')
            
            if not os.path.exists(train_path):
                raise Exception(f"Training directory not found: {train_path}")
            
            # Load training data
            for idx, emotion in enumerate(self.emotions):
                emotion_path = os.path.join(train_path, emotion)
                if os.path.exists(emotion_path):
                    for img_file in os.listdir(emotion_path):
                        img_path = os.path.join(emotion_path, img_file)
                        try:
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                img = cv2.resize(img, (self.img_size, self.img_size))
                                X_train.append(img)
                                y_train.append(idx)
                        except:
                            continue
            
            # Load testing data
            if os.path.exists(test_path):
                for idx, emotion in enumerate(self.emotions):
                    emotion_path = os.path.join(test_path, emotion)
                    if os.path.exists(emotion_path):
                        for img_file in os.listdir(emotion_path):
                            img_path = os.path.join(emotion_path, img_file)
                            try:
                                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                                if img is not None:
                                    img = cv2.resize(img, (self.img_size, self.img_size))
                                    X_test.append(img)
                                    y_test.append(idx)
                            except:
                                continue
            
            X_train = np.array(X_train).reshape(-1, self.img_size, self.img_size, 1) / 255.0
            X_test = np.array(X_test).reshape(-1, self.img_size, self.img_size, 1) / 255.0
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            
            print(f"Loaded {len(X_train)} training samples, {len(X_test)} testing samples")
            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def get_data_generators(self):
        """Create data generators with augmentation.

        Images are already normalized to [0, 1] by the loaders, so no
        rescale factor is applied here.
        """
        try:
            datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            return datagen
        except Exception as e:
            raise Exception(f"Error creating data generators: {str(e)}")
    
    def load_from_csv(self, csv_path):
        """Load data from FER2013 CSV format"""
        try:
            df = pd.read_csv(csv_path)
            pixels = df['pixels'].tolist()
            X = []
            for pixel_sequence in pixels:
                img = np.array(pixel_sequence.split(), dtype='float32')
                img = img.reshape(48, 48)
                X.append(img)
            X = np.array(X).reshape(-1, 48, 48, 1) / 255.0
            y = df['emotion'].values
            return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except Exception as e:
            raise Exception(f"Error loading from CSV: {str(e)}")
