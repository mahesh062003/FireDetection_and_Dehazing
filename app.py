from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import threading
from threading import Thread
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from configparser import ConfigParser
from image_dehazer import remove_haze
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('./Fire_Detection_model.h5')

config = ConfigParser()
config.read('config.ini')

Latitude = float(config.get('Location', 'Latitude'))
Longitude = float(config.get('Location', 'Longitude'))

SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_USERNAME = 'madhan.2105054@srec.ac.in'  # Replace with your email
SMTP_PASSWORD = 'madhan2004'  # Replace with your email password
RECIPIENT_EMAIL = 'madhan.2105054@srec.ac.in'  # Replace with recipient email

Alarm_Status = False
Fire_Reported = 0

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def start_detection(video_path):
    global Alarm_Status, Fire_Reported
    Alarm_Status = False
    Fire_Reported = 0

    video = cv2.VideoCapture(video_path)

    while True:
        (grabbed, frame) = video.read()
        if not grabbed:
            break

        HazeCorrectedImg, _ = remove_haze(frame,
                                           regularize_lambda=0.05,
                                           sigma=1,
                                           delta=1,
                                           showHazeTransmissionMap=False)

        prediction = detect_fire(HazeCorrectedImg)

        if prediction > 0.5:
            Fire_Reported += 1

        if Fire_Reported >= 1:
            if not Alarm_Status:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                screenshot_path = os.path.join(app.config['UPLOAD_FOLDER'], f"screenshot_{timestamp}.jpg")
                cv2.imwrite(screenshot_path, HazeCorrectedImg)
                threading.Thread(target=send_mail_function, args=(Latitude, Longitude, screenshot_path)).start()
                Alarm_Status = True

        cv2.imshow("Enhanced Output", HazeCorrectedImg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    video.release()

def send_mail_function(lat, lon, screenshot_path):
    try:
        print("Sending email...")  # Placeholder print statement
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.image import MIMEImage

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)

        google_maps_link = f"https://www.google.com/maps?q={lat},{lon}"
        message = f"Warning: A Fire Accident has been reported at the following location: {google_maps_link}"

        with open(screenshot_path, 'rb') as screenshot_file:
            screenshot_data = screenshot_file.read()

        msg = MIMEMultipart()
        msg.attach(MIMEText(message, 'plain'))

        image = MIMEImage(screenshot_data, name="screenshot.jpg")
        msg.attach(image)

        msg['Subject'] = "Fire Report"
        msg['From'] = SMTP_USERNAME
        msg['To'] = RECIPIENT_EMAIL

        server.sendmail(SMTP_USERNAME, RECIPIENT_EMAIL, msg.as_string())
        server.quit()

    except Exception as e:
        print(f"Exception in send_mail_function: {e}")

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (256, 256))
    normalized_frame = resized_frame / 255.0
    return normalized_frame

def detect_fire(frame):
    preprocessed_frame = preprocess_frame(frame)
    input_data = np.expand_dims(preprocessed_frame, axis=0)
    prediction = model.predict(input_data)
    return prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    page = 'home'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            if request.form.get('start_detection'):
                Thread(target=start_detection, args=(file_path,)).start()
                return render_template('index.html', page='upload', uploaded=True, detection_started=True)

    return render_template('index.html', page=page)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    page = 'upload'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            if request.form.get('start_detection'):
                Thread(target=start_detection, args=(file_path,)).start()
                return render_template('index.html', page='upload', uploaded=True, detection_started=True)

    return render_template('index.html', page=page)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
