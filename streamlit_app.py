import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import io

# Load the model
model_path = "model/telltale_model_v04.pt"  # Path to the YOLO model trained on tale-tell images
model = YOLO(model_path)

# Load telltale descriptions
tell_tale_descriptions_df = pd.read_csv('vehicle_telltale_rephrased.csv')
tell_tale_descriptions_df.set_index('Class_name', inplace=True)

# Function to predict and annotate image
def predict(image):
    # Perform inference
    results = model(image)

    # Extract predictions
    predictions = results[0].boxes.data.cpu().numpy()
    image_with_boxes = Image.fromarray(results[0].plot()[:, :, ::-1])

    # Prepare predictions dataframe
    df = pd.DataFrame(predictions, columns=["x_min", "y_min", "x_max", "y_max", "confidence", "class"])
    df["class"] = df["class"].astype(int)
    class_names = [model.names[int(cls)] for cls in df["class"]]
    df["class_name"] = class_names

    # Generate bounding box information
    bbox_info = ""
    detected_classes = df["class_name"].unique()
    for cls in detected_classes:
        bbox_info += f"- **{cls}**: {tell_tale_descriptions_df.loc[cls]['Class_definition']}\n\n"
        bbox_info += f"  - ***Root cause:*** {tell_tale_descriptions_df.loc[cls]['Root_cause']} \n\n"
        bbox_info += f"  - ***Possible Fix:*** {tell_tale_descriptions_df.loc[cls]['Possible_fix']} \n\n\n"

    return image_with_boxes, bbox_info

# Streamlit app setup
st.set_page_config(page_title="Telltale Assist", layout="wide")
st.markdown("<h1 style='text-align: center; margin-top: 0;'>Telltale Assist</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>*Your Partner in Vehicle Health and Safety.*</p>", unsafe_allow_html=True)

# File uploader or camera input
image_source = st.radio("Select Image Source:", ("Upload Image", "Capture from Camera"), horizontal=True)

if image_source == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

elif image_source == "Capture from Camera":
    captured_image = st.camera_input("Take a Picture")
    if captured_image is not None:
        image = Image.open(captured_image)

if 'image' in locals():
    # Run prediction
    with st.spinner("Processing image..."):
        pred_image, bbox_info = predict(image)

    # Display results side by side
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(pred_image, caption="Predicted Image with Bounding Boxes", use_container_width=True)

    with col2:
        st.markdown("#### Telltale Alerts and Corrective Actions")
        st.markdown(bbox_info)
