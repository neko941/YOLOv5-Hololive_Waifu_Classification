import os
import torch
import urllib
from PIL import Image
import streamlit as st
from pathlib import Path


def check_suffix(file='yolov5s.pt', suffix=('.pt',), msg=''):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


def check_file(file, suffix=''):
    # Search/download file (if necessary) and return path
    check_suffix(file, suffix)  # optional
    file = str(file)  # convert to str()
    if os.path.isfile(file) or not file:  # exists
        return file
    elif file.startswith(('http:/', 'https:/')):  # download
        url = file  # warning: Pathlib turns :// -> :/
        # '%2F' to '/', split https://url.com/file.txt?auth
        file = Path(urllib.parse.unquote(file).split('?')[0]).name
        if os.path.isfile(file):
            print(f'Found {url} locally at {file}')  # file already exists
        else:
            print(f'Downloading {url} to {file}...')
            torch.hub.download_url_to_file(url, file)
            assert Path(file).exists() and Path(file).stat(
            ).st_size > 0, f'File download failed: {url}'  # check
        return file


st.title("Hololive Waifu Classification")

image = st.text_input('Image URL', '')
st.info(
    'Images for quick tesing:\n \n \n'
    ' - https://i.imgur.com/tFZwWYw.jpg'
    '\n \n \n'
    ' - https://static.wikia.nocookie.net/omniversal-battlefield/images/b/bd/Council.jpg')
pretrained = st.selectbox('Select pre-trained', ('2022.11.01-YOLOv5x6_1280-Hololive_Waifu_Classification.pt', 'last.pt'))
imgsz = st.number_input(label='Image Size', min_value=None, max_value=None, value=1280, step=1)
conf = st.slider(label='Confidence threshold', min_value=0.0, max_value=1.0, value=0.25, step=0.01)
iou = st.slider(label='IoU threshold', min_value=0.0, max_value=1.0, value=0.45, step=0.01)
multi_label = st.selectbox('Multiple labels per box', (False, True))
agnostic = st.selectbox('Class-agnostic', (False, True))
amp = st.selectbox('Automatic Mixed Precision inference', (False, True))
max_det  = st.number_input(label='Maximum number of detections per image', min_value=None, max_value=None, value=1000, step=1)

if st.button('Excute'):
    with st.spinner('Loading the image...'):
        image_path = check_file(image)
        input_image = Image.open(image_path)
    with st.spinner('Loading the model...'):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join('pretrained', pretrained))
    with st.spinner('Updating configuration...'):
        model.conf = float(conf)
        model.max_det = int(max_det)
        model.iou = float(iou)
        model.agnostic = agnostic
        model.multi_label = multi_label
        model.amp = amp 
    with st.spinner('Predicting...'):
        results = model(input_image, size=int(imgsz))
    for img in results.render():
        st.image(img)
    st.write(results.pandas().xyxy[0])
    os.remove(image_path)
