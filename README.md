# Facial Expression Classification using CNN Deep Learning

## Project Overview

This project implements a comprehensive Facial Expression Classification system using Deep Convolutional Neural Networks (CNN). The system can classify human facial expressions into 7 different emotions and provides both web-based interfaces for image upload and real-time webcam detection.

## Features

- **Deep CNN Architecture**: Custom-built CNN with multiple convolutional blocks, batch normalization, dropout layers for robust feature extraction
- **7 Emotion Classes**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Image Upload & Prediction**: Upload images and get instant emotion classification with confidence scores
- **Live Webcam Detection**: Real-time facial expression detection using webcam feed
- **Real-time Face Detection**: OpenCV Haar Cascade classifier for accurate face localization
- **Data Augmentation**: Rotation, shifting, shearing, zooming, and horizontal flipping for better generalization
- **Comprehensive Training Pipeline**: Callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau), evaluation metrics
- **Visualization**: Training history plots, confusion matrix, classification reports
- **Exception Handling**: Robust error handling throughout the application
- **User-Friendly UI**: Clean Bootstrap-based web interface
- **Detailed Logging**: Application logging for debugging and monitoring

## Project Structure

```
Facial-Expression-Classification/
│
├── app.py                      # Flask web application
├── train.py                    # Model training script
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
├── app.log                     # Application logs (generated on run)
│
├── data/                       # Dataset directory
│   ├── train/                 # Training images (organize by emotion)
│   └── test/                  # Testing images (organize by emotion)
│
├── models/                     # Saved models and training artifacts
│   ├── best_model.h5          # Best model (highest validation accuracy)
│   ├── final_model.h5         # Final trained model
│   ├── training_history.png   # Training plots
│   ├── confusion_matrix.png   # Confusion matrix
│   └── training_report_*.json # Detailed metrics
│
├── notebooks/                  # Jupyter notebooks for experimentation
│
├── static/                     # Static files
│   ├── css/
│   │   └── style.css          # Custom styles
│   ├── js/
│   │   └── app.js             # Frontend JavaScript
│   └── uploads/               # Uploaded images (generated)
│
├── templates/                  # HTML templates
│   └── index.html             # Main web interface
│
└── utils/                      # Utility modules
    ├── __init__.py            # Utility functions and config
    ├── data_loader.py         # Data loading and preprocessing
    └── model.py               # CNN model architecture
```

## Dataset Preparation

This project supports two dataset formats:

### Option 1: Directory Structure (Recommended)
Organize your dataset as follows:
```
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

### Option 2: FER2013 CSV Format
Place `fer2013.csv` in the `data/` directory. The CSV should have columns: `emotion`, `pixels`, `Usage`.

**Note:** You need to add your own dataset. Popular sources:
- [FER2013 Dataset (Kaggle)](https://www.kaggle.com/datasets/msambare/fer2013)
- [AffectNet](http://mohammadmahoor.com/affectnet/)

## Installation & Setup

### Prerequisites
- Python 3.11–3.12 (required by the pinned TensorFlow 2.16.1 and kagglehub 1.0.3; newer Python versions do not have compatible TensorFlow wheels)
- Webcam for live detection (optional)
- CUDA-compatible GPU (optional, but recommended for faster training)

### Step 1: Clone or Download Project
Navigate to the project directory:
```bash
cd Facial-Expression-Classification
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Note:** TensorFlow installation may vary depending on your system. If you face issues, refer to the [TensorFlow installation guide](https://www.tensorflow.org/install).

### Step 4: Prepare Dataset
Either download the FER-2013 dataset automatically:
```bash
python data/data_set.py
```
This downloads the Kaggle dataset with `kagglehub` and copies its `train/` and `test/` folders into `data/`.

Or add your dataset manually to the `data/` folder following one of the formats mentioned above.

### Step 5: Train the Model
```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Build the CNN model
- Train with data augmentation
- Save best model to `models/best_model.h5`
- Generate training plots and evaluation reports

**Training time:** Depends on dataset size and hardware (can take 1-3 hours on GPU, longer on CPU)

### Step 6: Run the Web Application
```bash
python app.py
```

Open your browser and navigate to:
```
http://localhost:5000
```

## Usage Guide

### Image Upload Mode
1. Go to the "Upload Image" tab
2. Select an image file containing a face
3. Click "Classify Expression"
4. View results with predicted emotion, confidence score, and probability distribution for all emotions

### Live Webcam Mode
1. Go to the "Live Webcam" tab
2. Allow camera permissions when prompted
3. The system will detect faces in real-time and display predicted emotions with confidence scores

## Model Architecture

The CNN model consists of:

- **4 Convolutional Blocks**: Each with 2 Conv2D layers, BatchNormalization, ReLU activation, MaxPooling, and Dropout
- **Filter Sizes**: 64, 128, 256, 512 progressively increasing
- **Fully Connected Layers**: Two dense layers (1024, 512) with BatchNormalization, ReLU, and Dropout
- **Output Layer**: Softmax activation for 7-class classification
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Sparse Categorical Crossentropy

## Training Features

- **Data Augmentation**: Prevents overfitting by generating varied training samples
- **ModelCheckpoint**: Saves best model based on validation accuracy
- **EarlyStopping**: Stops training if validation loss doesn't improve for 15 epochs
- **ReduceLROnPlateau**: Reduces learning rate when validation loss plateaus
- **Stratified Split**: Ensures balanced class distribution in train/val sets

## Troubleshooting

### Model not found error
```
No trained model found. Please train the model first using train.py
```
**Solution:** Run `python train.py` first to train and save the model.

### Webcam not working
- Check if webcam is connected and not in use by another application
- Grant camera permissions to your browser
- Try changing camera index in `app.py` (change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`)

### Out of memory errors during training
- Reduce batch size in `utils/__init__.py` (change `BATCH_SIZE = 64` to `32` or `16`)
- Use smaller image size or reduce model complexity
- Use GPU if available

### No face detected
- Ensure image has clear, frontal face
- Good lighting conditions help
- Try different images or adjust face detection parameters

## Performance Tips

- **For Training**: Use GPU acceleration for faster training
- **For Inference**: CPU is sufficient for image upload and live detection
- **Dataset Quality**: More diverse, balanced data improves accuracy
- **Image Preprocessing**: Ensure consistent image quality

## Future Enhancements

- [ ] Add support for emotion intensity detection
- [ ] Implement face alignment for better accuracy
- [ ] Add more sophisticated face detectors (MTCNN, dlib)
- [ ] Create REST API endpoints
- [ ] Add model ensemble for better predictions
- [ ] Deploy to cloud platforms (Heroku, AWS, etc.)
- [ ] Add real-time emotion tracking graphs

## License

This project is created for educational and reference purposes.

## Acknowledgments

- FER2013 dataset creators
- OpenCV community
- TensorFlow/Keras team
- Flask framework

## Contact

For questions or issues, please refer to the project documentation or create an issue in the repository.

**Happy Learning!**
