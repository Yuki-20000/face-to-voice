import io
import logging

import cv2
import numpy as np
import soundfile as sf
import streamlit as st
from PIL import Image

from config import (
    ALLOWED_IMAGE_FORMATS,
    DEFAULT_TEXT,
    SAMPLE_RATE
)
from inference import FaceToVoiceEngine
from utils import validate_image, validate_text, reset_state

# -- Configure logging --
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# -- Setting --
st.set_page_config(
    page_title='Face-to-Voice AI',
    page_icon='🗣️',
    layout='centered',
    initial_sidebar_state='collapsed'
)

# -- Model Loading --
@st.cache_resource
def get_engine():
    return FaceToVoiceEngine()

try:
    engine = get_engine()
except Exception as e:
    st.error(f'Failed to load models: {e}')
    st.stop()

# -- Title --
st.title('🗣️ AI Face-to-Voice Generator')
st.markdown(
    '''
    **Upload a face image to predict and generate a voice.**  
    This AI analyzes facial features and demographics (Age, Gender) to synthesize a matching voice.
    '''
)
st.markdown('---')

# -- Face Image Input --
st.subheader('Face Image')
st.caption('Images are processed in-memory only and strictly not collected/stored.')
uploaded_file = st.file_uploader(
    'Upload Face Image',
    type=ALLOWED_IMAGE_FORMATS,
    on_change=reset_state
)

image = None
if uploaded_file is not None:
    # Validation check for image
    is_valid_img, img_err_msg = validate_image(uploaded_file)
    if not is_valid_img:
        st.error(f'Image Error: {img_err_msg}')
    else:
        # Display the uploaded image
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        st.image(image, width='stretch')

# -- Text Input --
st.subheader('Text to Speak')
input_text = st.text_area(
    'Enter text (English only)', 
    value=DEFAULT_TEXT,
    height=100,
    on_change=reset_state
)

# -- Generate Button --
generate_btn = st.button(
    'Generate Voice 🔊',
    type='primary',
    width='stretch',
    disabled=(image is None or not input_text)
)

if generate_btn:
    if image.mode != 'RGB':
        image = image.convert('RGB')

    img_array = np.array(image)

    # Convert to OpenCV format (BGR) for DeepFace
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Validation check for text
    is_valid, msg, clean_text = validate_text(input_text)
    if not is_valid:
        st.error(msg)
        reset_state()
        st.stop()

    with st.spinner('Generating voice...'):
        try:
            # Generation
            age, sex, wav = engine.generate_voice(
                img_array, 
                clean_text
            )

            # Save results to session state
            st.session_state['age'] = age
            st.session_state['sex'] = sex
            st.session_state['wav'] = wav

        except Exception as e:
            st.error(f'Generation Failed: {e}')
            reset_state()

if st.session_state.get('wav') is not None:
    # Display the predicted demographics
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label='Predicted Age', value=st.session_state.get('age'))
    with col2:
        st.metric(label='Predicted Gender', value=st.session_state.get('sex'))
    
    # Display audio
    st.audio(st.session_state.get('wav'), sample_rate=SAMPLE_RATE)

    # WAV Encoding
    buffer = io.BytesIO()
    sf.write(buffer, st.session_state.get('wav'), SAMPLE_RATE, format='WAV')
    buffer.seek(0)

    # Display download button
    st.download_button(
        label='Download WAV 📥',
        data=buffer,
        file_name='generated_voice.wav',
        mime='audio/wav',
        width='stretch'
    )