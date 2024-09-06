import os
import base64
import requests
import uuid
import json
import cv2 as cv
import numpy as np
from bs4 import BeautifulSoup
from io import BytesIO
from yunet import YuNet
import argparse
from tqdm import tqdm

def post_image_to_server(image_data, filename):
    # Generate a unique boundary
    boundary = f'dart-http-boundary-{uuid.uuid4()}'
    
    # Authorization token (base64 encoded)
    authorization_token = "Basic dXNlcjExMTpqZmtqZGZqazg4ODg4ODg4c3Mj"
    
    # JSON data to send with the image
    json_data = {
        "ref": "test",
        "ip": "1.1.1.3",
        "source": "android",
        "api_version": 3,
        "detect_gender": True
    }
    
    # Create the multipart form body
    body = (
        f'--{boundary}\r\n'
        f'Content-Type: application/octet-stream\r\n'
        f'Content-Disposition: form-data; name="form"; filename="{filename}"\r\n\r\n'
        + image_data.decode('latin-1') + '\r\n'
        f'--{boundary}\r\n'
        f'Content-Type: application/octet-stream\r\n'
        f'Content-Disposition: form-data; name="json"; filename="temp.json"\r\n\r\n'
        + json.dumps(json_data) + '\r\n'
        f'--{boundary}--\r\n'
    )
    
    # Set up the headers
    headers = {
        'user-agent': 'Dart/3.1 (dart:io)',
        'content-type': f'multipart/form-data; boundary={boundary}',
        'accept-encoding': 'gzip',
        'authorization': authorization_token,
        'host': 'app2.attractivenesstest.com'
    }
    
    # Make the request
    response = requests.post("http://app2.attractivenesstest.com", headers=headers, data=body.encode('latin-1'), verify=False)
    
    return response.text

def rotate_image_without_cropping(image, angle):
    # Get image dimensions
    h, w = image.shape[:2]
    # Calculate the center of the image
    center = (w // 2, h // 2)

    # Get rotation matrix
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)

    # Calculate the size of the new image that will fit after rotation
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # Adjust the rotation matrix to account for the translation
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # Perform the rotation with a transparent background
    rotated_image = cv.warpAffine(image, rotation_matrix, (new_w, new_h), flags=cv.INTER_CUBIC, borderValue=(0, 0, 0, 0))
    
    return rotated_image

def process_image(input_image_path, output_image_path):
    # Load model
    model = YuNet(modelPath='face_detection_yunet_2023mar.onnx',
                  inputSize=[320, 320],
                  confThreshold=0.9,
                  nmsThreshold=0.3,
                  topK=5000,
                  backendId=cv.dnn.DNN_BACKEND_OPENCV,
                  targetId=cv.dnn.DNN_TARGET_CPU)

    # Load input image
    image = cv.imread(input_image_path)
    if image is None:
        print(f"Error loading image: {input_image_path}")
        return

    print("Detecting faces...")

    h, w, _ = image.shape

    # Inference
    model.setInputSize([w, h])
    results = model.infer(image)

    # Detect faces
    num_faces = results.shape[0]
    print(f"Number of faces detected: {num_faces}")

    if num_faces == 0:
        print("No faces detected.")
        return

    errors = []

    # Draw results and save faces
    final_image = image.copy()

    # Determine the font size scaling factor (e.g., 1% of image height)
    font_scale = h / 1000  # Adjust this value based on your needs

    # Iterate over results with a progress bar
    for idx, det in tqdm(enumerate(results), total=num_faces, desc="Processing faces"):
        bbox = det[0:4].astype(np.int32)
        conf = det[-1]
        landmarks = det[4:14].astype(np.int32).reshape((5, 2))

        # Calculate angle of rotation based on eye landmarks
        eye_left = landmarks[0]
        eye_right = landmarks[1]
        dy = eye_right[1] - eye_left[1]
        dx = eye_right[0] - eye_left[0]
        angle = np.degrees(np.arctan2(dy, dx)) - 180

        # Expand bounding box by 30%
        x, y, w, h = bbox
        w, h = int(w * 1.3), int(h * 1.3)  # 30% expansion
        x, y = max(x - (w - bbox[2]) // 2, 0), max(y - (h - bbox[3]) // 2, 0)
        x2, y2 = min(x + w, final_image.shape[1]), min(y + h, final_image.shape[0])
        face = image[y:y2, x:x2]

        # Rotate face without cropping and with transparency
        rotated_face = rotate_image_without_cropping(face, angle)

        # Flip the rotated image 180 degrees
        rotated_face = cv.rotate(rotated_face, cv.ROTATE_180)

        # Create an alpha channel for transparency
        bgr = cv.split(rotated_face)
        if len(bgr) == 3:
            alpha_channel = np.ones(bgr[0].shape, dtype=bgr[0].dtype) * 255
            alpha_channel[np.all(rotated_face == 0, axis=-1)] = 0
            rotated_face_with_alpha = np.dstack([rotated_face, alpha_channel])  # Add alpha channel

            # Convert the rotated face to a JPEG in memory
            is_success, buffer = cv.imencode(".jpg", rotated_face_with_alpha)
            if is_success:
                image_bytes = BytesIO(buffer).getvalue()

                # Post each face to server and get response
                response_text = post_image_to_server(image_bytes, f'face_{idx}.jpg')
                
                # Parse the JSON response and extract 'score_adjusted'
                try:
                    response_json = json.loads(response_text)
                    if "score_adjusted" in response_json:
                        score_adjusted = round(float(response_json["score_adjusted"]), 2)  # Round to 2 decimal places
                    else:
                        # Collect error message
                        errors.append(f"\nError with face_{idx}:\n{response_text}")
                        score_adjusted = "N/A"
                except (json.JSONDecodeError, ValueError) as e:
                    # Collect error message
                    errors.append(f"\nError with face_{idx}:\n{response_text}\n{e}")
                    score_adjusted = "N/A"

                # Draw the score_adjusted on the final image with proportional font size
                cv.putText(final_image, str(score_adjusted), (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 3)

    # Print errors before saving the final image
    if errors:
        print("\n".join(errors))
    
    # Save final image
    cv.imwrite(output_image_path, final_image)
    print(f"Final image saved to {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an image with YuNet face detection.")
    parser.add_argument("--input", "-i", required=True, help="Path to the input image.")
    parser.add_argument("--output", "-o", default="faces_analyzed.jpg", help="Path to the output image (default: faces_analyzed.jpg).")

    args = parser.parse_args()

    process_image(args.input, args.output)
