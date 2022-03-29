import streamlit
import os
import pandas as pd
from streamlit_drawable_canvas import st_canvas
from PIL import Image

def main():
    selected_box = streamlit.sidebar.selectbox(
        'Choose one of the following',
        ('Welcome', 'Upload File', 'Drawing Time')
    )

    if selected_box == 'Welcome':
        welcome()
    if selected_box == 'Upload File':
        upload_file()
    if selected_box == 'Drawing Time':
        drawing_time()

def welcome():
    streamlit.title('Automatic drawing tool')

    streamlit.subheader('This is my thesis try it please')

    streamlit.image('1013024.png', use_column_width=True)

def upload_file():
    streamlit.title('Upload File Page')

    streamlit.subheader('Start trying this app by upload your own image, or using my image')

    uploaded_file = streamlit.file_uploader("Upload your image here", ['png', 'jpg'])

    if uploaded_file is not None:
        save_uploaded_data(uploaded_file)

def drawing_time():
    container = streamlit.container()

    # Specify canvas parameters in application
    stroke_width = container.slider("Stroke width: ", 1, 25, 3)
    stroke_color = container.color_picker("Stroke color hex: ")
    bg_color = container.color_picker("Background color hex: ", "#eee")
    bg_image = container.file_uploader("Background image:", type=["png", "jpg"])
    drawing_mode = container.selectbox(
        "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
    )
    realtime_update = container.checkbox("Update in realtime", True)

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        # height=150,
        drawing_mode=drawing_mode,
        key="canvas",
    )

    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        streamlit.image(canvas_result.image_data)
    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])  # need to convert obj to str because PyArrow
        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")
        streamlit.dataframe(objects)

def save_uploaded_data(uploadedFile):
    with open(os.path.join("gambar_upload", uploadedFile.name), "wb") as f:
        f.write(uploadedFile.getbuffer())

    return streamlit.success("Berhasil menyimpan file dengan nama {} di folder".format(uploadedFile.name))

if __name__ == "__main__":
    main()