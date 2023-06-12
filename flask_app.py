import argparse
import io
import os
from PIL import Image
import torch
from flask import Flask, render_template, request, redirect
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, redirect, flash, url_for
import urllib.request
from werkzeug.utils import secure_filename

app = Flask(__name__)

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
        detected_image = Image.open('static/images/image0.jpg')
        screenshot_image = Image.open('static/images/screenshot.jpg')
        detected_image_size = detected_image.size
        screenshot_image_size = screenshot_image.size
        merged_image = Image.new('RGB',(5*screenshot_image_size[0],5*screenshot_image_size[1]), (250,250,250))
        merged_image.paste(detected_image,(0,0))
        merged_image.paste(screenshot_image,(detected_image_size[0],0))
        merged_image.save("static/images/merged_image.jpg","JPEG")
        if prediction_result < predictions[0][1]:
            print("messy")
        else:
            print("clean")
        
        return redirect("static/images/merged_image.jpg")

    return render_template("index.html")

@app.route('/bookappt')
def bookappt():
    return render_template('bookappt.html')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)  # force_reload = recache latest code
    model.eval()

    app.run(host="0.0.0.0", port=args.port)
