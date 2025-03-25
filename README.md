# Number Plate Detection System

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.7.0-orange)
![Python](https://img.shields.io/badge/Python-3.9-blue)
![EasyOCR](https://img.shields.io/badge/EasyOCR-1.4.1-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5.5-red)

A real-time license plate detection and recognition system using deep learning with TensorFlow Object Detection API and EasyOCR.

![License Plate Detection Demo](https://github.com/faseehahmed26/Number-Plate-Detection/raw/main/Images/after.png)

## Features

- **Automatic License Plate Detection**: Identifies vehicle license plates in images and video streams
- **Text Recognition**: Extracts alphanumeric characters from detected license plates
- **Real-Time Processing**: Works with webcam feeds for live detection
- **Multi-Format Export**: TensorFlow.js and TFLite exports for web and mobile deployment
- **Confidence Filtering**: Removes low-confidence detections for improved accuracy

## Architecture

This project utilizes transfer learning with SSD MobileNet V2 FPNLite from TensorFlow Model Zoo, fine-tuned on a custom dataset of license plate images. The pipeline consists of:

1. **Detection Phase**: Locates license plates in the image using the trained model
2. **ROI Extraction**: Crops the license plate region from the larger image
3. **Text Recognition**: Uses EasyOCR to extract the alphanumeric text
4. **Post-Processing**: Filters results based on confidence scores and region characteristics
5. **Storage**: Saves results to CSV with images for further analysis

## Getting Started

### Prerequisites

- Python 3.9+
- TensorFlow 2.7.0
- CUDA-compatible GPU (recommended for real-time processing)

### Installation

```bash
# Clone the repository
git clone https://github.com/faseehahmed26/Number-Plate-Detection.git
cd Number-Plate-Detection

# Install dependencies
pip install -r requirements.txt
Usage
For webcam-based detection:
python app.py
For processing a single image:
# See notebook for detailed example
image_path = 'path_to_your_image.jpg'
detections = detect_fn(input_tensor)
text, region = ocr_it(image, detections, 0.6, 0.6)
print(f"Detected license plate: {text}")
```
Model Training
The model was trained using the TensorFlow Object Detection API with the following configuration:

Base Model: SSD MobileNet V2 FPNLite 320x320
Training Steps: 10,000
Batch Size: 4
Transfer Learning: Fine-tuning from COCO pre-trained weights

Project Structure

Tensorflow/: Contains the TensorFlow models and workspace
Detection_images/: Saved detection images and regions
app.py: Main application for real-time detection
*.ipynb: Jupyter notebooks for model training and testing

Future Improvements

Multi-plate detection in a single frame
Character segmentation for improved OCR accuracy
Integration with traffic monitoring systems
Speed optimization for edge devices

License
This project is licensed under the MIT License - see the LICENSE file for details.
