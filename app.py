import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash, session
import cv2
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image as keras_image
from torchvision import transforms
import torch
from facenet_pytorch import MTCNN

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'static/uploaded_files'
app.config['MODEL_FOLDER'] = 'models'

# Load the default model
model_path = os.path.join(app.config['MODEL_FOLDER'], 'IMAGE_DEEP_FAKE_MODEL-23.h5')

try:
    image_model = load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)

# Transformation steps
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_face(img):
    boxes, _ = mtcnn.detect(img)
    if boxes is not None:
        x, y, w, h = boxes[0].astype(int)
        face_img = img[y:h, x:w]
        return face_img, True
    return img, False  # If no face is detected, return the original image and False

def preprocess_frame(frame):
    face, detected = extract_face(frame)
    return keras_image.img_to_array(face), detected

def predict_image(file_path):
    try:
        print(f"Predicting image: {file_path}")
        img = keras_image.load_img(file_path)
        img = np.array(img)
        face_img, detected = extract_face(img)
        if not detected:
            return "No Face Detected", 0.0
        face_img = Image.fromarray(face_img)
        face_img = face_img.resize((224, 224))
        x = keras_image.img_to_array(face_img)
        x = np.expand_dims(x, axis=0)
        x /= 255.0
        y_pred = image_model.predict(x)
        confidence_score = y_pred[0][0]
        label = "Fake" if confidence_score >= 0.50 else "Real"
        print(f"Image prediction: {label} with confidence {confidence_score}")
        return label, confidence_score
    except Exception as e:
        print(f"Error in predict_image: {e}")
        raise

def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def predict_video(file_path):
    try:
        print(f"Predicting video: {file_path}")
        frames = extract_frames(file_path)
        predictions = []
        confidences = []
        face_detected = False
        for frame in frames:
            face_img, detected = extract_face(frame)
            if detected:
                face_detected = True
                face_img = Image.fromarray(face_img)
                face_img = face_img.resize((224, 224))
                x = keras_image.img_to_array(face_img)
                x = np.expand_dims(x, axis=0)
                x /= 255.0
                y_pred = image_model.predict(x)
                confidence_score = y_pred[0][0]
                label = "Fake" if confidence_score >= 0.50 else "Real"
                predictions.append(label)
                confidences.append(confidence_score)
        if not face_detected:
            return "No Face Detected", 0.0
        avg_confidence = np.mean(confidences)
        avg_prediction = max(set(predictions), key=predictions.count)
        return avg_prediction, avg_confidence
    except Exception as e:
        print(f"Error in predict_video: {e}")
        raise

def clear_uploaded_files():
    try:
        print("Clearing uploaded files.")
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("Uploaded files cleared.")
    except Exception as e:
        print(f"Error in clear_uploaded_files: {e}")
        raise

@app.route('/')
def index():
    print("Rendering index page.")
    return render_template('index.html')

@app.route('/scan')
def scan():
    print("Rendering scan page.")
    return render_template('scan.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    try:
        print("Uploading file.")
        clear_uploaded_files()
        session.clear()  # Clear session data
        if 'file' not in request.files:
            flash('No file part')
            return redirect(url_for('scan'))
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('scan'))
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            session['uploaded_file_path'] = file_path
            session['filename'] = file.filename
            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                session['file_type'] = 'image'
            elif file.filename.lower().endswith('.mp4'):
                session['file_type'] = 'video'
            else:
                flash('Unsupported file type')
                return redirect(url_for('scan'))
            flash(f'{file.filename} uploaded successfully.')
            print(f"File {file.filename} uploaded to {file_path}")
        return redirect(url_for('scan'))
    except Exception as e:
        print(f"Error in upload_file: {e}")
        flash('File upload failed')
        return redirect(url_for('scan'))

@app.route('/scan_now', methods=['POST'])
def scan_now():
    try:
        print("Scanning now.")
        file_path = session.get('uploaded_file_path')
        file_type = session.get('file_type')
        filename = session.get('filename')
        if file_path and file_type:
            print(f"Scanning file: {file_path} of type: {file_type}")
            if file_type == 'video':
                label, confidence = predict_video(file_path)
            elif file_type == 'image':
                label, confidence = predict_image(file_path)
            else:
                flash('Unsupported file type')
                return redirect(url_for('scan'))
            result_data = {
                'filename': filename,
                'label': label,
                'confidence': confidence,
                'file_type': file_type,
                'file_path': file_path
            }
            print(f"Scan result for {file_type} - {filename}: {label} with confidence {confidence}")
            session.clear()  # Clear session data
            return render_template('result.html', **result_data)
        else:
            flash('No file uploaded for scanning')
            return redirect(url_for('scan'))
    except Exception as e:
        print(f"Error in scan_now: {e}")
        flash('Scanning failed')
        return redirect(url_for('scan'))

@app.route('/upload_model', methods=['POST'])
def upload_model():
    try:
        print("Uploading model.")
        if 'model' in request.files:
            model_file = request.files['model']
            if model_file.filename == '':
                flash('No selected model file')
                return redirect(url_for('scan'))
            if model_file:
                model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_file.filename)
                model_file.save(model_path)
                load_models()  # Reload all models to update
                flash('Model loaded successfully')
                print(f"Model {model_file.filename} uploaded and loaded from {model_path}")
        return redirect(url_for('scan'))
    except Exception as e:
        print(f"Error in upload_model: {e}")
        flash('Model upload failed')
        return redirect(url_for('scan'))

if __name__ == '__main__':
    app.run(debug=True)
