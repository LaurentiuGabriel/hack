# Project Name
ULTRA SECRET HACKATHON PROJ

# Purpose
Identify damage in car pictures

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