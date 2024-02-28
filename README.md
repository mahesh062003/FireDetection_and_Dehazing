# Fire Detection And Dehazing 

This is a Flask web application for fire detection in videos. The application allows users to upload a video file, and the system detects the presence of fire in the video using a pre-trained deep learning model. If a fire is detected, an email notification with a screenshot and location information is sent to a specified recipient.

## Table of Contents
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Configuration](#configuration)

## Getting Started

### Prerequisites
Make sure you have the following installed:
- Python (version >= 3.6)
- Flask
- OpenCV
- TensorFlow
- image_dehazer
- Werkzeug

## Installation
Install the required Python packages using:
```bash
pip install -r requirements.txt
```
## File Structure

This project implements a fire detection system with the following components:

- **app.py:** Flask web application
- **Fire_Detection_model.h5:** Pre-trained deep learning model for fire detection
- **image_dehazer.py:** Module for haze removal in images
- **uploads/:** Directory to store uploaded video files and screenshots
- **config.ini:** Configuration file for location information
- **readme.md:** Documentation file

## Usage

1. **Run the Application:**
   - Execute the Flask web application using `app.py`.

2. **Upload Video Files:**
   - Visit the home page to upload video files.

3. **Optional: Start Detection:**
   - Optionally, check the "Start Detection" checkbox to initiate fire detection.

4. **Submit and Process:**
   - Submit the form, and the system will process the video.

5. **Detection Results:**
   - If a fire is detected, an email notification will be sent, and the enhanced video output will be displayed.

## Configuration

- Modify the `config.ini` file to update location information as needed.
