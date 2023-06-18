# Project Name
ULTRA SECRET HACKATHON PROJ

# Purpose
Identify damage in car pictures. Moreover, the API has a transcribe endpoint, which accepts an audio file and converts it into text. Then, using prompt engineering, we send the text to GPT 3.5 turbo model, and the model returns a JSON with the details that we want for insurance. The following screenshot shows how GPT 3.5 responds with our prompt:

<img width="854" alt="image" src="https://github.com/LaurentiuGabriel/hack/assets/17592225/8b18ff62-4170-495c-a2aa-d546f50a5ec2">

# Requirements
- Tensorflow
- Flask
- PIL
- matplotlib
- requests
- scipy
- torch
- pandas

# How to run this code
```
git clone https://github.com/LaurentiuGabriel/hack.git
cd hack
python flask_app.py
open in localhost
```

# Types of output
YOLOv5 will return image if damaged with bounding boxes namely of 4 possibilities 
- Scratch
- Glass broken
- Deformation
- Broken

DMG16 will return output in String format in the cmd as either "messy" or "clean"
