# Face Attractiveness Detection

This project provides a Python script to analyze the attractiveness of individuals in a group photo using the reverse-engineered [Attractiveness Test](https://play.google.com/store/apps/details?id=com.rsapps.attractiveness_test_flutter) Android API. The script utilizes the YuNet model for face detection and performs image processing to evaluate and display attractiveness scores for each detected face.

## Features

- Detects faces in a group image.
- Rotates and aligns faces for better accuracy.
- Uploads individual face images to the Attractiveness Test API for analysis.
- Draws attractiveness scores directly on the original image.

## Disclaimer
I am not affiliated with attractivenesstest.com in any way. The use of their API in this project is for educational purposes only. I am not responsible for any issues or consequences that may arise from using their services.

## Requirements
```bash
opencv-python
numpy
requests
tqdm
```

## Installation

To get started, clone this repository and install the required Python packages.

### Clone the Repository

```bash
git clone https://github.com/Andrei-Iosifescu123/face-attractiveness-analysis.git
cd face-attractiveness-analysis
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

### Known Issues
The server can't evaluate some faces and returns `{"error":"Face detected, but error during age detection 'with_padding'"}`.
