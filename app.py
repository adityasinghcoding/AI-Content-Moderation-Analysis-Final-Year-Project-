from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import tensorflow as tf
import easyocr
import cv2
import re
import speech_recognition as sr
import joblib
import os
import numpy as np
import nltk
# nltk.download('vader_lexicon')
from tensorflow.keras.preprocessing import image
from nltk.sentiment import SentimentIntensityAnalyzer
import moviepy.editor as mp

app = Flask(__name__, template_folder='template')

# Load the trained model
model = joblib.load('toxicity_classifier.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
img_model = load_model('IMG_MODEL.299x299.h5')
video_model = load_model('IMG_MODEL.299x299.h5')
violence_model = load_model('VIOLENCE_DETECTION.h5')

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict.html')
def predict():
    return render_template('predict.html')

def detect_violence(image_path):
    img=image.load_img(image_path,target_size=(224,224))
    img_array=image.img_to_array(img)
    img_array=np.expand_dims(img_array,axis=0)
    img_array/=255.0
    
    prediction=violence_model.predict(img_array)
    
    if prediction[0][0] >= 0.5:
            return True;
    else:
        return False;

def transcribe_audio(audio_path):   
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        transcribed_text = recognizer.recognize_google(audio)
   
    return transcribed_text

# Function to detect toxicity in an audio file using the trained model

def detect_toxicity_in_audio(audio_path):
    # Transcribe audio to text
    transcribed_text = transcribe_audio(audio_path)
   
    # Preprocess text similar to the training data
    transcribed_text = re.sub(r'[^a-zA-Z0-9\s]', '', transcribed_text)
   
    # Convert text to numerical features using TF-IDF vectorization
    audio_features = vectorizer.transform([transcribed_text])
   
    # Make predictions
    toxicity_result = model.predict(audio_features)
   
    return transcribed_text, toxicity_result

# Function to extract text from an image using OCR

def extract_text_from_image(image_path):
    
    img = cv2.imread(image_path)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(img_gray, (5,5), 0)
    
    reader = easyocr.Reader(['en'])
    result = reader.readtext(blurred_img)
    extracted_text = ' '.join([entry[1] for entry in result])
    
    return extracted_text

def detect_toxicity_in_image(image_path):
    
    extracted_text = extract_text_from_image(image_path)

    extracted_text = re.sub(r'[^a-zA-Z0-9\s]', '', extracted_text)
   
    # Convert text to numerical features using TF-IDF vectorization
    image_features = vectorizer.transform([extracted_text])
   
    toxicity_result = model.predict(image_features)
    
    return extracted_text, toxicity_result

class_labels = ["drawings", "hentai", "neutral", "porn", "sexy"]

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    return img_array

def classify_image(img_path, threshold=16):
    img_array = preprocess_image(img_path)
    predictions = img_model.predict(img_array)[0]  # Binary classification (0 or 1)
    
    # Calculate percentages
    percentages = [round(prob * 100, 2) for prob in predictions]
    
    # Map percentages to class labels
    result = {label: percentage for label, percentage in zip(class_labels, percentages)}
    
    # Display class labels and percentages
    print("Class\t\tPercentage")
    print("-----------------------")
    for label in class_labels:
        print(f"{label}\t\t{result[label]}%")

    # Check if any of the labels exceed the threshold
    for label in ["sexy", "porn", "hentai"]:
        if result[label] > threshold:
            return True  # Image can't be published

    return False  # Image can be published

# def preprocess_video_frame(frame):
#     frame = cv2.resize(frame, (299, 299))
#     frame = frame / 255.0  # Normalize pixel values between 0 and 1
#     return frame

# def predict_video_frame(frame):
#     preprocessed_frame = preprocess_video_frame(frame)
#     preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)  # Add batch dimension
#     predictions = video_model.predict(preprocessed_frame)[0]  # Get predictions for the first (and only) frame
#     return predictions

# def video_classification(video_path, threshold=16):
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0
#     nudity_detected = False
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         frame_count += 1
#         if frame_count % (3 * cap.get(cv2.CAP_PROP_FPS)) == 0:  # Extract frame every 3 seconds
#             frame_path = os.path.join(UPLOAD_FOLDER, f"frame_{frame_count}.jpg")
#             cv2.imwrite(frame_path, frame)  # Save frame to disk
#             nudity_detected = classify_image(frame_path, threshold)
            
#             if nudity_detected:
#                 break
    
#     cap.release()
    
#     return nudity_detected

def extract_frames_from_video(video_path, frame_interval=3):
    cap = cv2.VideoCapture(video_path)
    frames = []

    if not cap.isOpened():
        print("Error opening video file.")
        return frames

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Extract a frame every 'frame_interval' seconds
        if frame_count % int(cap.get(cv2.CAP_PROP_FPS) * frame_interval) == 0:
            frames.append(frame)

    cap.release()
    return frames

# Function to preprocess the image array for the model
def preprocess_image_from_array(img_array):
    img_array = cv2.resize(img_array, (299, 299))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    return img_array

# Function to classify frames
def classify_video_frames(video_path):
    frames = extract_frames_from_video(video_path)

    if not frames:
        print("No frames extracted.")
        return

    predictions_for_frames = []
    for frame in frames:
        # Convert OpenCV BGR format to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Classify the frame using the existing function
        img_array = preprocess_image_from_array(frame_rgb)
        predictions = video_model.predict(img_array)[0]
        percentages = [round(prob * 100, 2) for prob in predictions]
        result = {label: percentage for label, percentage in zip(class_labels, percentages)}
        predictions_for_frames.append(result)
        
        return calculate_average_percentages(predictions_for_frames)

# Function to calculate average percentages
def calculate_average_percentages(predictions_list):
    if not predictions_list:
        return None

    num_frames = len(predictions_list)
    num_classes = len(class_labels)

    # Initialize sums for each class
    class_sums = {label: 0 for label in class_labels}

    # Accumulate percentages for each class
    for frame_result in predictions_list:
        for label, percentage in frame_result.items():
            class_sums[label] += percentage

    # Calculate average percentages
    average_percentages = {label: class_sums[label] / num_frames for label in class_labels}
    return average_percentages

def text_from_video(video_path):
    
    # Load the video file
    clip = mp.VideoFileClip(video_path)

    # Extract audio
    audio_path = "audio.wav"
    clip.audio.write_audiofile(audio_path)

    # Convert audio to text using Google Speech Recognition
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = r.record(source)
        try:
            text = r.recognize_google(audio_data)
            toxicity_result = model.predict(text)
            return text, toxicity_result
        except sr.UnknownValueError:
            print("Unable to understand the audio !")
        except sr.RequestError as e:
            print()                

def detect_toxicity_in_text(text):
    text_processed = re.sub(r'[^a-zA-Z0-9\s]', '',text)
    test_tfidf = vectorizer.transform([text_processed])  
    toxicity_result = model.predict(test_tfidf)
    
    sia = SentimentIntensityAnalyzer()

# Analyze sentiment for each test text and classify based on sentiment score
    sentiment_predictions = []
    sentiment_score = sia.polarity_scores(text)
    if sentiment_score['compound'] >= 0.05:  # Positive sentiment threshold
        sentiment_predictions.append('positive')
    elif sentiment_score['compound'] <= -0.05:  # Negative sentiment threshold
        sentiment_predictions.append('negative')
    else:
        sentiment_predictions.append('neutral')
    
    return text_processed, toxicity_result , sentiment_predictions

model = joblib.load('toxicity_classifier.pkl')

@app.route('/detect_toxicity', methods=['POST'])
def detect_toxicity():
    if request.method == 'POST':
        text = request.form.get('text')
        uploaded_file = request.files.get('file')
        if uploaded_file and uploaded_file.filename != '':
            file_ext = os.path.splitext(uploaded_file.filename)[1]
            if file_ext.lower() == '.wav':
                # Handle audio file
                audio_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
                uploaded_file.save(audio_path)
                audio_text, toxicity_result = detect_toxicity_in_audio(audio_path)
                return jsonify({'text': audio_text, 'toxicity_result': 'Toxic' if toxicity_result == 1 else 'Non-Toxic'})
            elif file_ext.lower() in ['.jpg', '.jpeg', '.png']:
                # Handle image file
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
                uploaded_file.save(image_path)
                image_text, text_result = detect_toxicity_in_image(image_path)
                violence_result = detect_violence(image_path)
                can_publish = classify_image(image_path)
                if not image_text:
                    return jsonify({'text': 'No Text Detected', 'toxicity_result': 'Cannot be published' if can_publish or violence_result else 'Can be published'})
                else:
                    return jsonify({'text': image_text, 'toxicity_result': 'Cannot be published' if text_result == 1 or (can_publish or violence_result) else 'Can be Published'})
            elif file_ext.lower() == '.mp4':
                # Handle video file
                video_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
                uploaded_file.save(video_path)
                nudity_detected = classify_video_frames(video_path)
                text_processed, toxicity_result = text_from_video(video_path)
                if not text_processed:
                        return jsonify({'text': 'No Text Detected', 'toxicity_result': 'Nudity detected in video' if nudity_detected >= 16 else 'Can be published'})
                else:
                    return jsonify({'text': text_processed, 'toxicity_result': 'Cannot be published' if nudity_detected >= 16 or toxicity_result else 'Can be published'})
            else:
                return jsonify({'error': 'Unsupported file format'})
        elif text:
            # Handle text input
            text_processed, toxicity_result, sentiment_result = detect_toxicity_in_text(text)
            return jsonify({'text': text_processed , 'toxicity_result': 'Toxic' if toxicity_result == 1 or sentiment_result == 'negative' else 'Non-Toxic'})
        else:
            return jsonify({'Error': 'No text or file provided'})

        
if __name__=='__main__':
    app.run(debug=True)
