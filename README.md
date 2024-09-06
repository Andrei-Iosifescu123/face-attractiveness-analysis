# Face Attractiveness Detection

This project provides a Python script to analyze the attractiveness of individuals in a group photo using the reverse-engineered [Attractiveness Test](https://play.google.com/store/apps/details?id=com.rsapps.attractiveness_test_flutter) Android API. The script utilizes the YuNet model for face detection and performs image processing to evaluate and display attractiveness scores for each detected face.

## Features

- Detects faces in a group image.
- Rotates and aligns faces for better accuracy.
- Uploads individual face images to the Attractiveness Test API for analysis.
- Draws attractiveness scores directly on the original image.

## Requirements
```bash
opencv-python-headless
numpy
requests
beautifulsoup4
tqdm
```

## Installation

To get started, clone this repository and install the required Python packages.

### Clone the Repository

```bash
git clone https://github.com/yourusername/face_attractiveness_detection.git
cd face_attractiveness_detection
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Usage
```bash
main.py [-h] --input INPUT [--output OUTPUT]
```
#### Example Usage:
```bash
python3 main.py --input examples/example1.jpg
```
