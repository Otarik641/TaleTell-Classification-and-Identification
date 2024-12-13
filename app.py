import gradio as gr
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from io import BytesIO

# Load the model
model_path = "model/telltale_model.pt"  # model's path
model = YOLO(model_path)

# Class definitions (update with your classes and definitions)
tell_tale_descriptions_df = pd.read_csv('vehicle_telltale_rephrased.csv')
tell_tale_descriptions_df.index = tell_tale_descriptions_df['Class_name']


# class_definitions = {
#     "Service Indicator" : "In case of Power Train Sensors & Actuators failure then this Amber indicator will glow. When there is high severity then Red indicator will glow.",
#     "Immobilizer" : "This lamp comes on when the system disables vehicle start if the original key is not used.", 
#     "LV Battery charging" : "If it remains ‘ON’ while the vehicle is running, it indicates that the battery is not getting charged. Switch off all unnecessary electrical equipment and get the problem attended at TATA authorized Service center.",
#     "Airbag status" : "This lamp comes on when ignition is switched ‘ON’ and goes ‘OFF’ in approx. 4 seconds. If it continuously remains on or blinks then contact the TATA MOTORS",
#     "Park Brake / Brake Fluid Low / EBD malfunction" : "If it remains ‘ON’, it indicates Brake fluid level is low or ABS/EBD system has a fault.",
#     "EPAS" : "Illuminates when there is a fault in the",
#     "Driver Seat Belt Indicator" : "If seat belt is not fastened then Telltale will be ON as initial warning",
#     "Key Not Detected" : "This lamp comes on when the Valid Smart key is not detected inside the vehicle.",
#     "Press Brake Pedal to Start vehicle" : "This lamp comes on with IGN ON till user presses the Brake pedal to start the vehicle.",
#     "ABS Indicator" : "Illuminates continuously if there is any malfunction in ABS.",
#     "Speed limit warning indicator" : "If vehicle speed crosses 120 kmph, the speed limit warning indicator flashes",
#     "TPMS" : "This symbol comes ON and blink for 4 second if Tyre Pressure or Temperature is LOW/HIGH,",
#     "HHC warning lamp" : "If continuously on then HHC, system is in fault condition.",
#     "HDC Warning lamp" : "If continuously ON then HDC system is at fault condition",
#     "HV Critical Alert" : "When there is high severity then Red indicator will glow.",
#     "Charging Fail Indicator" : "This symbol is displayed when the vehicle is not getting charged even if the",
#     "Charger Connected" : "This symbol lights up as soon as the charger is connected for charging the battery",
#     "Motor High Temperature" : "This symbol lights up when the temperature of the motor is higher, and motor becomes hot.",
#     "Battery High Temperature" : "This symbol lights up when the temperature of the battery is higher, and battery becomes hot.",
#     "Limp Home Mode" : "This symbol indicates the vehicle gone into limited performance mode.",
#     "High Voltage (HV) Alert" : "This symbol lights up the voltage of the battery is too high and cause damage.",
#     "Electronic Stability Control (ESC)" : "If continuously ON then ESP system is at fault condition",
#     "Co-Driver Seat Belt Indicator" : "Co-Passenger is not wearing the seat belt",
#     "Zero Charge/ Low Charge" : "Illuminate if battery charge become too low",
#     "ACC" : "This symbol indicate that vehicle is in Accessory mode",
#     "Malfunction Indication Lamp (MIL)" : "It is a tell-tale that a computerized engine-management system used to indicate a malfunction or problem with the vehicle",
#     "Low Oil Pressure indicator" : "it indicates a fault in the electrical circuit / lubrication system",
#     "AIB Warning" : "Its AIB Warning, need to update",
#     "Check Engine Lamp" : "This lamp comes on continuously if a fault arises in Engine Management System.",
#     "ESP OFF Indicator" : "This feature monitor the Electronic Stability Program (ESP) input and informs the driver about ESP status.",
#     "AMT Fault / DCA Fault" : "Illuminates continuously when there is a fault in Automated Manual Transmission system.",
#     "High Coolant Temperature" : "If the engine overheats due to higher coolant temperatures, this indicator blinks along with an audible buzzer.",
#     "Charging Indicator" : "This symbol is displayed when your vehicle is getting charged."
# }

# Function to process the image and return predictions
def predict(image):
    # Perform inference
    results = model(image)

    # Extract predictions
    predictions = results[0].boxes.data.cpu().numpy()  # Convert to numpy array
    image_with_boxes = Image.fromarray(results[0].plot()[:, :, ::-1])

    # convert pil image from BGR to RGB
    # image_with_boxes = image_with_boxes

    # Prepare predictions dataframe
    df = pd.DataFrame(predictions, columns=["x_min", "y_min", "x_max", "y_max", "confidence", "class"])
    df["class"] = df["class"].astype(int)
    class_names = [model.names[int(cls)] for cls in df["class"]]
    df["class_name"] = class_names

    # Generate bounding box information
    bbox_info = ""
    detected_classes = df["class_name"].unique()
    for cls in detected_classes:
        bbox_info += f"- **{cls}**: {tell_tale_descriptions_df.loc[cls]['Class_definition']}\n\n  - ***Root cause:*** {tell_tale_descriptions_df.loc[cls]['Root_cause']} \n\n  - ***Possible Fix:*** {tell_tale_descriptions_df.loc[cls]['Possible_fix']} \n\n\n\n"

    return image_with_boxes, bbox_info


# Define Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=[
        gr.Image(type="pil", label="Predicted Image with Bounding Boxes"),
        # gr.Text(label="Prediction Dataframe"),
        gr.Markdown(label="Definition of Predicted Classes"),
    ],
    title="Tale Tell Image Identification",
    description="Upload an image to classify tale tell signs along with bounding boxes and class definitions.",
    live = True,
    cache_examples=False
)

# Launch Gradio App
if __name__== "__main__":
    demo.launch()
