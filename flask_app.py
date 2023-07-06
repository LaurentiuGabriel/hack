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
import fitz 
from PyPDF2 import PdfReader
from PIL import ImageChops
from PIL import ImageEnhance

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
EXIF_DATE_FORMAT = "%Y:%m:%d %H:%M:%S"
ERROR_LEVEL_ANALYSIS = True
HISTOGRAM_ANALYSIS = True
METADATA_CHECK = True

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

def check_embedded_objects(file_path):
    pdf_file = PdfReader(file_path)
    if "/AcroForm" in pdf_file.trailer["/Root"]:
        return False
    return True

def check_fonts(file_path):
    doc = fitz.open(file_path)
    fonts = set()
    for i in range(len(doc)):
        for font in doc.get_page_fonts(i):
            fonts.add(font[3])  # the font name is at index 3

    suspicious_fonts = {"UnknownFont"}  # add suspicious font names here

    for font in fonts:
        if font in suspicious_fonts:
            return False
    return True

def ela_analysis(filename):
    basename, extension = os.path.splitext(filename)
    resaved = basename + '.resaved.jpg'
    ela = basename + '.ela.png'  
    im = Image.open(filename)
    im.save(resaved, 'JPEG', quality=95)
    resaved_im = Image.open(resaved)
    ela_im = ImageChops.difference(im, resaved_im)
    return ela_im

def get_im_extreme(ela_im):
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    return max_diff    

def detect_fraud_through_ela(max_diff, threshold=10):
    if max_diff > threshold:
        return True
    else: 
        return False

def check_pdf(file_path, date):
    pdf_file = open(file_path, 'rb')
    read_pdf = PdfReader(pdf_file)
    info = read_pdf.metadata
    print(f"PDF Metadata: {info}")

    # Check the creation date
    if info.creation_date and info.creation_date.date() < date:

        print("The PDF date is not within the policy date: " + info.creation_date.date() + "<" + date)
        return False
    if info.modification_date != None:
        if info.creation_date != info.modification_date:
            print("The creation date is different than the modification date.")
            return False

    # Check if the PDF is encrypted
    if read_pdf.is_encrypted:
        print("The PDF should not be encrypted.")
        return False
    
    if(check_embedded_objects(file_path) == False):
        print("The PDF contains embedded AcroForms")
        return False

    if(check_fonts(file_path) == False):
        print("The PDF contains irregular fonts!")
        return False

    
    
    return True


def check_image_tampering(file_path, threshold):
    img = Image.open(file_path)
    exif_data = img._getexif()

    datetime_original = None
    datetime_digitized = None


    if exif_data is not None:
        for tag, value in exif_data.items():
            tagname = PIL.ExifTags.TAGS.get(tag, tag)
            if tagname == 'Software':
                print(tagname + ": " + value)
                global METADATA_CHECK
                METADATA_CHECK = False
                return True
            if tagname == 'Copyright':
                print(f'Copyright info: {value}')  # Print copyright info
                if value:
                    
                    METADATA_CHECK = False
                    print("tag copyright is here!")
                    return True  # Image may have a copyright, return True

            # Consistency check: Compare DateTimeOriginal and DateTimeDigitized
            if tagname == 'DateTimeOriginal':
                datetime_original = value
            elif tagname == 'DateTimeDigitized':
                datetime_digitized = value

        # If both DateTimeOriginal and DateTimeDigitized exist, check if they are the same
        if datetime_original and datetime_digitized and datetime_original != datetime_digitized:
            METADATA_CHECK = False
            print('DateTimeOriginal and DateTimeDigitized are not the same!')
            return True

    # Histogram analysis
    img_hist = img.histogram()

    # Separate channels
    r = img_hist[0:256]
    g = img_hist[256:256*2]
    b = img_hist[256*2:256*3]

    # Check for spikes in histogram
    hist_threshold = threshold * img.size[0] * img.size[1]  # threshold set to 10% of the total pixels
    for hist_channel in [r, g, b]:
        if max(hist_channel) > hist_threshold:
            print(str(max(hist_channel)) + " > " + str(hist_threshold))
            print("Possible image tampering detected: Spike in histogram")
            global HISTOGRAM_ANALYSIS
            HISTOGRAM_ANALYSIS = False
            return True
    
    ela_im = ela_analysis(file_path)
    error_value = get_im_extreme(ela_im)
    if(detect_fraud_through_ela(error_value)):
        print("Possible image tampering identified through ELA")
        global ERROR_LEVEL_ANALYSIS
        ERROR_LEVEL_ANALYSIS = False 
        return True

    # If no suspicious spikes in histogram, return False (no tampering detected)
    return False

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

        
        file_type = filename.rsplit('.', 1)[1].lower()

        if file_type == 'pdf':
            tampering_result = check_pdf(file_path, start_date)
            if(tampering_result == False):
                return jsonify({'error': 'The PDF file seems to have been tampered with.'}), 200
        
            return jsonify({
            'filename': filename,
            'policyStartDate': policy_start_date,
            'policyEndDate': policy_end_date,
            'country': country,
            'status': "The document seems to be the original one",
            "details": {"ErrorLevelAnalysisPassed": ERROR_LEVEL_ANALYSIS, "HistogramAnalysisPassed": HISTOGRAM_ANALYSIS, "MetaDataCheckPassed": METADATA_CHECK }
        })

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
                if tag == PIL.ExifTags.TAGS.get('DateTimeOriginal') or tag == PIL.ExifTags.TAGS.get('DateTime') or tag == 306 :
                    photo_date = datetime.strptime(value, EXIF_DATE_FORMAT).date()
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
        if check_image_tampering(file_path, 0.0245):
            
            return jsonify({'error': 'The image seems to have been tampered with.', "details": {"ErrorLevelAnalysisPassed": ERROR_LEVEL_ANALYSIS, "HistogramAnalysisPassed": HISTOGRAM_ANALYSIS, "MetaDataCheckPassed": METADATA_CHECK }}), 200
   

        # Process the data...
        return jsonify({
            'filename': filename,
            'policyStartDate': policy_start_date,
            'policyEndDate': policy_end_date,
            'country': country,
            'status': "The document seems to be the original one",
            "details": {"ErrorLevelAnalysisPassed": ERROR_LEVEL_ANALYSIS, "HistogramAnalysisPassed": HISTOGRAM_ANALYSIS, "MetaDataCheckPassed": METADATA_CHECK }
        })

    return jsonify({'error': 'Allowed file types are png, jpg, jpeg, pdf'}), 400


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

    return chat_with_gpt(text)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Damage Detective")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # force_reload = recache latest code
    model.eval()

    app.run(host="0.0.0.0", port=args.port)
