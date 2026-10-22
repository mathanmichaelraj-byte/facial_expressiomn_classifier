"""
CNN Model Architecture for Facial Expression Classification
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

class EmotionCNN:
    def __init__(self, input_shape=(48, 48, 1), num_classes=7):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        """Build a deep CNN architecture"""
        try:
            model = Sequential()
            
            model.add(Input(shape=self.input_shape))
            
            # Block 1
            model.add(Conv2D(64, (3, 3), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Conv2D(64, (3, 3), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            
            # Block 2
            model.add(Conv2D(128, (5, 5), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Conv2D(128, (5, 5), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            
            # Block 3
            model.add(Conv2D(256, (3, 3), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Conv2D(256, (3, 3), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            
            # Block 4
            model.add(Conv2D(512, (3, 3), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Conv2D(512, (3, 3), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            
            # Fully Connected Layers
            model.add(Flatten())
            model.add(Dense(1024))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            
            model.add(Dense(512))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            
            model.add(Dense(self.num_classes, activation='softmax'))
            
            self.model = model
            return model
            
        except Exception as e:
            raise Exception(f"Error building model: {str(e)}")
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        try:
            if self.model is None:
                self.build_model()
            optimizer = Adam(learning_rate=learning_rate)
            self.model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            print("Model compiled successfully")
            return self.model
        except Exception as e:
            raise Exception(f"Error compiling model: {str(e)}")
    
    def get_callbacks(self, model_path='models/best_model.h5'):
        """Get training callbacks"""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            callbacks = [
                ModelCheckpoint(
                    model_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            return callbacks
        except Exception as e:
            raise Exception(f"Error creating callbacks: {str(e)}")
    
    def load_trained_model(self, model_path):
        """Load a trained model"""
        try:
            if not os.path.exists(model_path):
                raise Exception(f"Model file not found: {model_path}")
            self.model = load_model(model_path)
            print(f"Model loaded from {model_path}")
            return self.model
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def summary(self):
        """Print model summary"""
        if self.model:
            return self.model.summary()
        else:
            print("Model not built yet")
