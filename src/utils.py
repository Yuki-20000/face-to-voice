import re

import streamlit as st
from better_profanity import profanity
from PIL import Image, UnidentifiedImageError

from config import SESSION_STATE_KEYS, MAX_TEXT_LENGTH

profanity.load_censor_words()

def reset_state():
    '''
    Reset the generation results in session state.
    '''
    keys = SESSION_STATE_KEYS
    for k in keys:
        st.session_state.pop(k, None)

def validate_image(uploaded_file):
    '''
    Check if the uploaded image is valid.
    Args:
        uploaded_file: Input file

    Returns:
        is_valid (bool)
        error_msg
    '''

    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)

        return True, ''

    except UnidentifiedImageError:
        return False, 'Uploaded file is not a valid image.'
    except Exception as e:
        return False, f'Image validation failed: {str(e)}'

def sanitize_numbers(text):
    '''
    Splits long sequences of numbers (15+ digits) with hyphens to ensure digit-by-digit reading.
    '''
    return re.sub(r'\d{15,}', lambda x: '-'.join(x.group()), text)

def validate_text(text):
    '''
    Check if the input text is valid.
    Args:
        text: Input text

    Returns:
        is_valid (bool)
        error_msg
        cleaned_text
    '''
    # Empty / Whitespace Check
    if not text or not text.strip():
        return False, 'Input text is required.', ''

    # Cut numbers if it is too long
    text = sanitize_numbers(text)

    # Cleaning
    cleaned_text = re.sub(r'\s+', ' ', text).strip()

    # Length Check
    if len(cleaned_text) > MAX_TEXT_LENGTH:
        return False, f'Text is too long (Max {MAX_TEXT_LENGTH} chars).', ''

    # Minimum length check
    if len(cleaned_text) < 2:
        return False, 'Text is too short. Please enter at least 2 characters.', ''
    
    # Character Whitelist Check
    allowed_pattern = r'^[a-zA-Z0-9.,!?\'" \-\n]+$'
    if not re.match(allowed_pattern, cleaned_text):
        return False, "Invalid characters detected. Only English letters, numbers, and basic punctuation (.,!?'\"-) are allowed.", ''

    # Minimum content check
    if not re.search(r'[a-zA-Z]', cleaned_text):
        return False, 'Text must contain at least one letter.', ''
    
    # Profanity Check
    if profanity.contains_profanity(cleaned_text):
        return False, 'Text contains inappropriate content.', ''
    
    return True, '', cleaned_text