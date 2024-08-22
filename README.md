# Eye and Iris Detection with MediaPipe

## Description

This project uses MediaPipe, OpenCV, and NumPy to detect eye and iris positions in real-time from a webcam feed. It leverages MediaPipe's Face Mesh model to track facial landmarks and estimate the eye aspect ratio (EAR) to determine if the eyes are open or closed. Additionally, it detects the iris position within the eye and visualizes the results on the video feed.

## Features

- Real-time eye and iris detection using MediaPipe Face Mesh
- Calculation of Eye Aspect Ratio (EAR) to determine eye status (open/closed)
- Iris position detection and visualization
- Adjustable EAR threshold for customized sensitivity

## Installation

To set up this project on your local machine, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/eye-iris-detection.git
   cd eye-iris-detection
