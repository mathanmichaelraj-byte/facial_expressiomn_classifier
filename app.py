"""
Flask Web Application for Facial Expression Classification
Features:
- Live webcam detection
- Image upload and prediction
- Real-time face detection
- Confidence scores display
- Exception handling
"""
import os
import sys
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import EmotionConfig, preprocess_image, detect_faces, load_haarcascade
from utils.model import EmotionCNN

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global variables
face_cascade = None
model = None
config = EmotionConfig()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load face detection cascade and emotion classification model"""
    global face_cascade, model
    try:
        logger.info("Loading face cascade classifier...")
        face_cascade = load_haarcascade()
        logger.info("Face cascade loaded successfully")
        
        logger.info("Loading emotion classification model...")
        cnn = EmotionCNN(input_shape=(config.IMG_SIZE, config.IMG_SIZE, 1), 
                          num_classes=config.NUM_CLASSES)
        
        # Try to load best model first, then final model
        model_paths = ['models/best_model.h5', 'models/final_model.h5']
        model_loaded = False
        for model_path in model_paths:
            if os.path.exists(model_path):
                model = cnn.load_trained_model(model_path)
                model_loaded = True
                logger.info(f"Model loaded from {model_path}")
                break
        
        if not model_loaded:
            raise Exception("No trained model found. Please train the model first using train.py")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

def predict_emotion(face_img):
    """Predict emotion from face image"""
    try:
        processed, _ = preprocess_image(face_img, config.IMG_SIZE)
        predictions = model.predict(processed, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][emotion_idx] * 100)
        emotion_label = config.get_emotion_label(emotion_idx)
        return emotion_label, confidence, predictions[0]
    except Exception as e:
        logger.error(f"Error predicting emotion: {str(e)}")
        raise

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load models on startup
try:
    load_models()
except Exception as e:
    logger.warning(f"Models not loaded on startup: {str(e)}")
    logger.warning("Make sure to train the model using train.py before using the web app")

@app.route('/')
def index():
    """Render main page"""
    try:
        return render_template('index.html', emotions=config.EMOTIONS)
    except Exception as e:
        logger.error(f"Error rendering index: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and prediction"""
    try:
        if model is None or face_cascade is None:
            return jsonify({
                'error': 'Model not loaded. Train the model and restart the server.'
            }), 503
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in request'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read and process image
            img = cv2.imread(filepath)
            if img is None:
                return jsonify({'error': 'Invalid image file'}), 400
            
            # Detect faces
            faces, gray = detect_faces(img, face_cascade)
            
            if len(faces) == 0:
                return jsonify({'error': 'No face detected in the image. Please upload a clear photo with a face.'}), 400
            
            results = []
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                emotion_label, confidence, all_probs = predict_emotion(face_roi)
                
                # Get all emotion probabilities
                prob_dict = {config.EMOTIONS[i]: float(all_probs[i] * 100) 
                             for i in range(len(config.EMOTIONS))}
                
                results.append({
                    'emotion': emotion_label,
                    'confidence': round(confidence, 2),
                    'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                    'probabilities': prob_dict
                })
            
            return jsonify({
                'success': True,
                'filename': filename,
                'num_faces': len(faces),
                'results': results
            })
        else:
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, gif, bmp'}), 400
            
    except Exception as e:
        logger.error(f"Error in upload: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def generate_frames():
    """Generate frames for live webcam stream"""
    camera = None
    try:
        if model is None or face_cascade is None:
            logger.error("Cannot start stream: model or face cascade not loaded")
            return

        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            logger.error("Cannot access webcam")
            return
        
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            # Detect faces
            faces, gray = detect_faces(frame, face_cascade)
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                try:
                    emotion_label, confidence, _ = predict_emotion(face_roi)
                    # Draw rectangle and label
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    label = f"{emotion_label}: {confidence:.1f}%"
                    cv2.putText(frame, label, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except Exception as e:
                    logger.warning(f"Could not classify face: {str(e)}")
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        logger.error(f"Error in video stream: {str(e)}")
    finally:
        if camera is not None:
            camera.release()

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    try:
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Error in video feed: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/api/status')
def status():
    """API endpoint for status check"""
    try:
        return jsonify({
            'status': 'running',
            'model_loaded': model is not None,
            'face_cascade_loaded': face_cascade is not None,
            'emotions': config.EMOTIONS
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    try:
        logger.info("Starting Facial Expression Classification Web App...")
        logger.info("Access the app at http://localhost:5000")
        debug = os.environ.get('FLASK_DEBUG', '0').lower() in ('1', 'true', 'yes')
        app.run(debug=debug, host='0.0.0.0', port=5000, threaded=True)
    except Exception as e:
        logger.error(f"Error starting app: {str(e)}")
        sys.exit(1)
