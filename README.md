# Project Name
ClaimsDetective

# Purpose
The Claims Detective API is an innovative solution designed to combat fraud in motor insurance claims. By analyzing user-uploaded images and decoding voice messages, it meticulously inspects vehicle damage, verifies picture authenticity and timeframe, ultimately enhancing the accuracy and reliability of claim assessments.

# Types of output for Identifying Damage
The model will return image if damaged with bounding boxes namely of 4 possibilities 
- Scratch
- Glass broken
- Deformation
- Broken

DMG16 will return output in String format in the cmd as either "messy" or "clean"

# 🌐 API Documentation
## 📥 POST /formdata
This endpoint allows for the upload of image files (png, jpg, jpeg) along with associated form data.

### Parameters:
#### file: Image file to be uploaded
#### policyStartDate: The start date of the policy (in format YYYY-MM-DD)
#### policyEndDate: The end date of the policy (in format YYYY-MM-DD)
#### country: The country of the policy
### Responses:
#### 200 OK: On successful operation, returns JSON containing filename, policy start and end dates, and country.
#### 400 Bad Request: When an error occurs, returns a JSON with an error message.

## 🔮 POST /transcribe
This endpoint allows for the upload of audio files for transcription and analysis.

### Parameters:
#### file: Audio file to be uploaded
### Responses:
#### 200 OK: On successful operation, returns a chatbot response.
#### 400 Bad Request: When an error occurs, returns a JSON with an error message.

## 🎯 POST /damage
This endpoint accepts an image file for damage detection.

### Parameters:
#### file: Image file to be uploaded
### Responses:
#### 200 OK: On successful operation, redirects to a rendered image with detected damage.
#### 302 Found: Redirects back to the request URL if the file is not found.