import argparse
import io
import os
from PIL import Image
import PIL.ExifTags
from geopy.geocoders import Nominatim
import torch
from flask import Flask, render_template, request, redirect
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from datetime import datetime
import speech_recognition as sr
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, redirect, jsonify
import urllib.request
from werkzeug.utils import secure_filename
import openai

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
EXIF_DATE_FORMAT = "%Y:%m:%d %H:%M:%S"

app = Flask(__name__)

openai.api_key = os.environ["OPENAI"]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_country(latitude, longitude):
    geolocator = Nominatim(user_agent="myGeocoder")
    location = geolocator.reverse([latitude, longitude], exactly_one=True)
    address = location.raw['address']
    country = address.get('country', '')
    return country

def chat_with_gpt(message):
    response = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",
              messages=[{"role": "system", "content": 'You are an world class automated API that handle text input. I want you to analyze the message that you get and check out if in the message there is any mention about victims, damaged hood, damaged windows, damaged doors, damaged tyres, damaged trunk or total loss. Respond only with a JSON file containing all these elements, and a numeric counter as a value for each JSON property. Do not respond using any text, like greetings or any of that stuff. Just respond with that JSON. REMEMBER, just the JSON, nothing else! Absolutely no introduction, no other text, just the JSON!!!'},
                        {"role": "user", "content": "This is the text message: " + message}
              ])
    return response["choices"][0]["message"]["content"]

@app.route("/formdata", methods=["POST"])
def post_formdata():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['file']

    # If user does not select file, browser submits an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    if file and allowed_file(file.filename):
        # secure the filename before storing it directly on the filesystem
        filename = secure_filename(file.filename)
        file_path = os.path.join('static/save/', filename)
        file.save(file_path)

        policy_start_date = request.form.get('policyStartDate')
        policy_end_date = request.form.get('policyEndDate')
        country = request.form.get('country')

        # Parse dates and check validity
        try:
            start_date = datetime.strptime(policy_start_date, "%Y-%m-%d").date()
            end_date = datetime.strptime(policy_end_date, "%Y-%m-%d").date()
            if start_date >= end_date:
                return jsonify({'error': 'Policy start date must be before end date.'}), 400
        except ValueError:
            return jsonify({'error': 'Incorrect date format, should be YYYY-MM-DD.'}), 400

        # Fetch the EXIF data from the image
        img = PIL.Image.open(file_path)
        exif_data = img._getexif()
        

        if exif_data is not None:
            photo_date = None
            lat_ref = None
            lon_ref = None
            lat_val = None
            lon_val = None
            
            for tag, value in exif_data.items():
                print(tag)
                if tag == PIL.ExifTags.TAGS.get('DateTimeOriginal') or tag == PIL.ExifTags.TAGS.get('DateTime') or tag == 306 :
                    photo_date = datetime.strptime(value, EXIF_DATE_FORMAT).date()
                    print(photo_date)
                elif tag == PIL.ExifTags.TAGS.get('GPSInfo'):
                    for t, v in getattr(value, 'items', lambda: {})():
                        if t == PIL.ExifTags.GPSTAGS.get('GPSLatitudeRef'):
                            lat_ref = v
                        elif t == PIL.ExifTags.GPSTAGS.get('GPSLongitudeRef'):
                            lon_ref = v
                        elif t == PIL.ExifTags.GPSTAGS.get('GPSLatitude'):
                            lat_val = v
                        elif t == PIL.ExifTags.GPSTAGS.get('GPSLongitude'):
                            lon_val = v
            if photo_date is not None:
                if photo_date < start_date or photo_date > end_date:
                    return jsonify({'error': 'Photo date is not within policy period.'}), 400
            if lat_ref and lon_ref and lat_val and lon_val:
                latitude = lat_val[0] + lat_val[1] / 60 + lat_val[2] / 3600
                if lat_ref == 'S':
                    latitude = -latitude
                longitude = lon_val[0] + lon_val[1] / 60 + lon_val[2] / 3600
                if lon_ref == 'W':
                    longitude = -longitude
                photo_country = get_country(latitude, longitude)
                if photo_country != country:
                    return jsonify({'error': 'Photo location does not match policy country.'}), 400

        # Process the data...
        return jsonify({
            'filename': filename,
            'policyStartDate': policy_start_date,
            'policyEndDate': policy_end_date,
            'country': country
        })

    return jsonify({'error': 'Allowed file types are png, jpg, jpeg.'}), 400


@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        uploaded_file = request.files["file"]
        if not uploaded_file:
            return
        img_bytes = uploaded_file.read()
        input_image = Image.open(io.BytesIO(img_bytes))
        detection_results = model(input_image)  # inference
        detection_results.render()  # updates results.ims with boxes and labels
        Image.fromarray(detection_results.ims[0]).save("static/images/image0.jpg")
        filename=uploaded_file.filename
        resized_image = image.load_img(filename, target_size=(224, 224))
        preprocessed_image = image.img_to_array(resized_image)
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        preprocessed_image = preprocess_input(preprocessed_image)
        damage_model = load_model('dmg_model.h5')
        predictions = damage_model.predict(preprocessed_image)
        prediction_result = predictions[0][0]
        if prediction_result < predictions[0][1]:
            print("messy")
        else:
            print("clean")
        
        return redirect("static/images/image0.jpg")

    return render_template("index.html")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400

    file = request.files['file']
    recognizer = sr.Recognizer()

    with sr.AudioFile(file) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        print(text)

    return jsonify(chat_with_gpt(text))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Damage Detective")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # force_reload = recache latest code
    model.eval()

    app.run(host="0.0.0.0", port=args.port)
