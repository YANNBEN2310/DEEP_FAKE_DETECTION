import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash, session
import cv2
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from facenet_pytorch import MTCNN

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'static/uploaded_files'
app.config['MODEL_FOLDER'] = 'models'

# Load the default model
video_model_path = os.path.join(app.config['MODEL_FOLDER'], 'model_97_acc_100_frames_FF_data.pt')
image_model_path = os.path.join(app.config['MODEL_FOLDER'], 'model_97_acc_100_frames_FF_data.pt')
video_model = None
image_model = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        seq_length, c, h, w = x.shape  # Update this line to match the dataset output
        x = x.view(seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(1, seq_length, 2048)  # Update this line to add batch dimension
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

video_model = Model(num_classes=2).to(device)

def load_models():
    global video_model, image_model
    if os.path.exists(video_model_path):
        try:
            print(f"Loading video model from {video_model_path}")
            video_model.load_state_dict(torch.load(video_model_path, map_location=device))
            video_model.eval()
            print(f"Video model loaded from {video_model_path}")
        except Exception as e:
            print(f"Error loading video model: {e}")
            video_model = None
    else:
        print(f"Error: The video model file does not exist at {video_model_path}")
        video_model = None

    if os.path.exists(image_model_path):
        try:
            print(f"Loading image model from {image_model_path}")
            image_model = torch.load(image_model_path, map_location=device)
            image_model.eval()
            print(f"Image model loaded from {image_model_path}")
        except Exception as e:
            print(f"Error loading image model: {e}")
            image_model = None
    else:
        print(f"Error: The image model file does not exist at {image_model_path}")
        image_model = None

load_models()

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

class VideoDataset(Dataset):
    def __init__(self, video_path, transform, sequence_length=60):
        self.video_path = video_path
        self.transform = transform
        self.sequence_length = sequence_length
        self.frames = self.extract_frames()
        self.detector = MTCNN()

    def extract_frames(self):
        frames = []
        cap = cv2.VideoCapture(self.video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        frames = []
        for i, frame in enumerate(self.frames):
            if i % (len(self.frames) // self.sequence_length) == 0:
                boxes, _ = self.detector.detect(frame)
                if boxes is not None and len(boxes) > 0:
                    x1, y1, x2, y2 = [int(coord) for coord in boxes[0]]
                    frame = frame[y1:y2, x1:x2]
                frames.append(self.transform(frame))
            if len(frames) == self.sequence_length:
                break
        frames = torch.stack(frames)
        return frames  # Return shape (sequence_length, c, h, w)

im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def predict_video(video_path):
    print(f"Predicting video: {video_path}")
    dataset = VideoDataset(video_path, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    confidences = []
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            _, logits = video_model(batch)
            probs = nn.functional.softmax(logits, dim=1)
            conf, pred = torch.max(probs, 1)
            predictions.append(pred.item())
            confidences.append(conf.item())
            print(f"Batch prediction: {pred.item()}, confidence: {conf.item()}")
    avg_confidence = np.mean(confidences)
    avg_prediction = np.argmax(np.bincount(predictions))
    label = 'Fake' if avg_prediction == 1 else 'Real'
    print(f"Video prediction: {label}, average confidence: {avg_confidence}")
    return label, avg_confidence

def predict_image(file_path):
    global image_model
    try:
        print(f"Predicting image: {file_path}")
        if image_model is None:
            print("Error: Image model is not loaded.")
            return "Error", 0.0
        img = Image.open(file_path).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)
        y_pred = image_model(img)
        confidence_score = y_pred[0][1].item()
        label = "Fake" if confidence_score >= 0.50 else "Real"
        print(f"Image prediction: {label} with confidence {confidence_score}")
        return label, confidence_score
    except Exception as e:
        print(f"Error in predict_image: {e}")
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
