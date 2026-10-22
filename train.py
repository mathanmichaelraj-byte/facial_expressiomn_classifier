"""
Training script for Facial Expression Classification CNN
"""
import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.model import EmotionCNN
from utils.data_loader import DataLoader
from utils import EmotionConfig, plot_training_history, plot_confusion_matrix, save_training_report

def main():
    try:
        print("=" * 60)
        print("FACIAL EXPRESSION CLASSIFICATION - TRAINING")
        print("=" * 60)
        
        # Configuration
        config = EmotionConfig()
        print(f"Emotions: {config.EMOTIONS}")
        print(f"Image Size: {config.IMG_SIZE}x{config.IMG_SIZE}")
        print(f"Batch Size: {config.BATCH_SIZE}")
        print(f"Epochs: {config.EPOCHS}")
        print("=" * 60)
        
        # Load data
        print("\n[1/5] Loading Dataset...")
        data_loader = DataLoader(data_path='data', img_size=config.IMG_SIZE)
        
        # Try to load from directory structure first
        if os.path.exists(os.path.join('data', 'train')):
            X_train, y_train, X_test, y_test = data_loader.load_from_directory()
        elif os.path.exists('data/fer2013.csv'):
            print("Loading from fer2013.csv...")
            X_train, X_test, y_train, y_test = data_loader.load_from_csv('data/fer2013.csv')
        else:
            print("\nWARNING: No dataset found!")
            print("Please place your dataset in one of the following formats:")
            print("1. data/train/[emotion]/ and data/test/[emotion]/ (folders)")
            print("2. data/fer2013.csv")
            print("\nEmotions needed: angry, disgust, fear, happy, neutral, sad, surprise")
            return
        
        # Split training data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Testing samples: {len(X_test)}")
        
        # Build and compile model
        print("\n[2/5] Building CNN Model...")
        cnn = EmotionCNN(input_shape=(config.IMG_SIZE, config.IMG_SIZE, 1), 
                          num_classes=config.NUM_CLASSES)
        model = cnn.build_model()
        cnn.compile_model(learning_rate=config.LEARNING_RATE)
        model.summary()
        
        # Get callbacks
        print("\n[3/5] Setting up callbacks...")
        callbacks = cnn.get_callbacks('models/best_model.h5')
        
        # Data augmentation
        print("\n[4/5] Training Model...")
        datagen = data_loader.get_data_generators()
        train_generator = datagen.flow(X_train, y_train, batch_size=config.BATCH_SIZE)
        
        # Train model
        history = model.fit(
            train_generator,
            steps_per_epoch=max(1, len(X_train) // config.BATCH_SIZE),
            epochs=config.EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        model.save('models/final_model.h5')
        print("\nFinal model saved to models/final_model.h5")
        
        # Evaluate model
        print("\n[5/5] Evaluating Model...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Generate reports
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=config.EMOTIONS))
        
        # Plot results
        plot_training_history(history, 'models/training_history.png')
        plot_confusion_matrix(y_test, y_pred_classes, config.EMOTIONS, 'models/confusion_matrix.png')
        save_training_report(history, y_test, y_pred_classes, config.EMOTIONS)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Files saved:")
        print("- models/best_model.h5 (best validation accuracy)")
        print("- models/final_model.h5 (final epoch)")
        print("- models/training_history.png")
        print("- models/confusion_matrix.png")
        print("- models/training_report_*.json")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
