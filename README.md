# Telltale Image Identification

This project is a Streamlit application for identifying and classifying telltale signs from vehicle instrument cluster images. It uses a YOLO-based object detection model to annotate uploaded or captured images with bounding boxes and provides detailed information about detected telltale signs, including definitions, root causes, and possible fixes.

## Features
- **Upload Images**: Upload an image from your local device for prediction.
- **Capture from Camera**: Use your webcam to capture an image for analysis.
- **Real-time Predictions**: Automatically processes the image to detect telltale signs and provide annotations.
- **Detailed Explanations**: Displays definitions, root causes, and fixes for each detected class.
- **User-friendly Interface**: Easy-to-use Streamlit interface with a responsive layout.

## Requirements
Ensure the following are installed:

- Python 3.8+
- Required Python libraries (listed in `requirements.txt`):
  ```
  streamlit
  pandas
  pillow
  ultralytics
  ```
- A trained YOLO model (`telltale_model.pt`).
- A CSV file (`vehicle_telltale_rephrased.csv`) containing telltale class definitions, root causes, and fixes.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repository/telltale-image-identification.git
   cd telltale-image-identification
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the YOLO model (`telltale_model.pt`) and `vehicle_telltale_rephrased.csv` are in the `model` directory and the project root, respectively.

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run streamlit_telltale.py
   ```

2. Open the application in your browser at the URL provided by Streamlit (e.g., `http://localhost:8501`).

3. Select an image source:
   - Upload an image file (JPEG or PNG).
   - Capture an image using your webcam.

4. View results:
   - The processed image with bounding boxes will appear alongside detailed information about the detected telltale classes.

## Project Structure
```
project-directory/
├── model/
│   └── telltale_model.pt      # Trained YOLO model
├── vehicle_telltale_rephrased.csv  # Telltale class definitions, root causes, fixes
├── streamlit_telltale.py      # Streamlit application script
├── requirements.txt           # Required Python libraries
└── README.md                  # Project documentation
```

## Deployment
To host this app for free, use **Streamlit Community Cloud**:

1. Push your code to a GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with your GitHub account.
3. Deploy the app by linking your GitHub repository and specifying the `streamlit_telltale.py` file.

## License
This project is licensed under the MIT License.

## Acknowledgments
- YOLO for object detection.
- Streamlit for the interactive interface.

