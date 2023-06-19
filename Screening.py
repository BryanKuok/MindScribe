import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_drawable_canvas import st_canvas
import numpy as np
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

def get_prediction(image): 
    np.set_printoptions(suppress=True)

# Load the model
    model = load_model("keras_Model.h5", compile=False)

# Load the labels
    class_names = open("labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
    image = Image.open(image).convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
    image_array = np.asarray(image)

# Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)

def drawing():
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
    point_display_radius = st.sidebar.slider("Point display radius:", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke color hex:")
    bg_color= st.sidebar.color_picker("Background colour hex:", "#eee")
    bg_image = st.sidebar.file_uploader("Upload Sample Clock Draw :", type = ["png", "jpg"])
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    canvas_result = st_canvas(
        fill_color = "rgba(225, 165, 0, 0.3)", 
        stroke_width = stroke_width, 
        stroke_color = stroke_color, 
        background_color = bg_color,
        background_image = Image.open(bg_image) if bg_image else None,
        update_streamlit = realtime_update, 
        width = 300, 
        point_display_radius = point_display_radius, 
        key = "canvas", 
    )

    image = Image.fromarray(canvas_result.image_data)
    
    screen = st.button("Submit")

    if canvas_result.image_data is not None and screen:
        get_prediction(image)


selected = option_menu(
    menu_title = None, 
    options = ["Patient's Particular", "Command Clock", "Copy Clock"],
    icons = ["1-circle", "2-circle", "3-circle"],
    orientation = "horizontal", 
)

if selected == "Patient's Particular":
    with st.form(key = "form1"):
        name = st.text_input(label = "Enter your name"),
        age = st.slider(label = "Enter your age", min_value = 0, max_value = 100),
        submit = st.form_submit_button(label = "Submit personal particulars")

if selected == "Command Clock":
    st.title("Please draw a clock, put in all the numbers in the clock, and set the time to 10 after 11")
    drawing()
if selected == "Copy Clock":
    st.title("Please replicate the clock shown below")
    image = Image.open('clock-normal.png')
    st.image(image)
    drawing()