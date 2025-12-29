import numpy as np
import pandas as pd
import torch
from deepface import DeepFace
from sklearn.preprocessing import normalize
from styletts2.tts import StyleTTS2
from tensorflow.keras.models import load_model

from config import MALE_MODEL_PATH, FEMALE_MODEL_PATH, MALE_VOICE_SCALE, FEMALE_VOICE_SCALE

class FaceToVoiceEngine:
    '''
    A pipeline class to handle the end-to-end process of:
    1. Extracting face embeddings.
    2. Combining face features with Age.
    3. Mapping the combined vector to a voice style.
    4. Generating speech audio.
    '''

    def __init__(self):
        # Setup Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load ArcFace Model
        try:
            DeepFace.build_model('ArcFace', task='facial_recognition')
            DeepFace.build_model('Age', task='facial_attribute')
            DeepFace.build_model('Gender', task='facial_attribute')
        except Exception as e:
            raise Exception(f'Warning: DeepFace loading failed. ({e})')

        # Load StyleTTS 2 Model
        try:
            self.tts = StyleTTS2()
        except Exception as e:
            raise Exception(f'Warning: StyleTTS 2 loading failed. ({e})')

        # Load FaceToVoice Models
        try:
            if not MALE_MODEL_PATH.exists():
                raise FileNotFoundError(f'Male model not found: {MALE_MODEL_PATH}')
            if not FEMALE_MODEL_PATH.exists():
                raise FileNotFoundError(f'Female model not found: {FEMALE_MODEL_PATH}')
        
            self.ftv_male = load_model(MALE_MODEL_PATH, compile=False, safe_mode=False)
            self.ftv_female = load_model(FEMALE_MODEL_PATH, compile=False, safe_mode=False)
        except Exception as e:
            raise Exception(f'Warning: FaceToVoice loading failed. ({e})')

    def _extract_face_embedding(self, img_array):
        '''
        Extract ArcFace embedding.
        Args:
            img_array: An array of input face image

        Returns:
            np.ndarray: A face embedding vector representing the input face.
        '''

        try:
            embedding_objs = DeepFace.represent(
                img_path=img_array,
                model_name='ArcFace',
                detector_backend='opencv',
                enforce_detection=True
            )
            return embedding_objs[0]['embedding']
    
        except Exception as e:
            if 'Face could not be detected' in str(e):
                raise Exception('Face could not be detected. Please upload a clear image containing a visible face.')
            
            raise Exception(f'DeepFace extraction failed: {e}')
        
    def _extract_attributes(self, img_array):
        '''
        Analyzes the face image to predict Demographics.
        Args:
            img_array: An array of input face image

        Returns:
            pred_age: Predicted age by DeepFace.
            pred_sex: Predicted sex by DeepFace.
                Either 'Man' or 'Woman'.
        '''

        try:
            # Predict Age, Gender
            analysis = DeepFace.analyze(
                img_path=img_array,
                actions=['age', 'gender'],
                detector_backend='opencv',
                enforce_detection=True,
                silent=True
            )
            result = analysis[0] if isinstance(analysis, list) else analysis
            
            # Age
            pred_age = int(result['age'])
            pred_age = max(0, pred_age)  # Set to zero if negative
            
            # Gender
            pred_sex = result['dominant_gender']
            
            return pred_age, pred_sex

        except Exception as e:
            if 'Face could not be detected' in str(e):
                raise Exception('Face could not be detected. Please upload a clear image containing a visible face.')
            
            raise Exception(f'Attribute prediction failed: {e}')
        
    def _prepare_input_vector(self, face_embedding, age):
        '''
        Preprocess and concatenate features.
        Args:
            face_embedding: The face embedding vector.
            age: The subject's age.

        Returns:
            np.ndarray: A combined vector that is used for predicting voice.
        '''

        # Face (L2 Norm)
        face_vec = np.array(face_embedding, dtype=np.float32).reshape(1, -1)
        face_vec_norm = normalize(face_vec, norm='l2')

        # Age (Categorize and One-Hot Encoding)
        age_bins = [0, 20, 30, 40, 50, 60, np.inf]
        age_labels = ['0-19', '20-29', '30-39', '40-49', '50-59', '60+']
        age_category = pd.cut([age], bins=age_bins, labels=age_labels, right=False)[0]

        age_idx = age_labels.index(age_category)
        age_onehot = np.zeros((1, 6), dtype=np.float32)
        age_onehot[0, age_idx] = 1.0

        # Concatenate
        return np.hstack([face_vec_norm, age_onehot])

    def _predict_style_vector(self, sex, input_vec):
        '''
        Predict the style vector based on sex.
        Args:
            input_vec: A combined vector
            sex: The subject's sex.

        Returns:
            PyTorch Tensor: A predicted vector that is used for generating voice.
        '''

        if sex == 'Man':
            pred = self.ftv_male.predict(input_vec, verbose=0)
            pred_scaled = pred * MALE_VOICE_SCALE
        elif sex == 'Woman':
            pred = self.ftv_female.predict(input_vec, verbose=0)
            pred_scaled = pred * FEMALE_VOICE_SCALE
        else:
            raise ValueError(f'Invalid sex value: {sex}.')
        
        # Numpy Array -> PyTorch Tensor
        return torch.tensor(pred_scaled, dtype=torch.float32).to(self.device)

    def generate_voice(self, img_array, text):
        '''
        Generate audio from face and attributes.
        Args:
            img_array: The input face image.
            text: The text to speak (English).

        Returns:
            wav: Generated audio waveform (sample rate: 24kHz).
        '''

        # Extract face embedding
        face_emb = self._extract_face_embedding(img_array)

        # Extract attributes
        age, sex = self._extract_attributes(img_array)

        # Prepare vector
        input_vec = self._prepare_input_vector(face_emb, age)
        
        # Predict Style Vector
        speaker_embedding = self._predict_style_vector(sex, input_vec)
        
        # Generate Audio
        try:
            wav = self.tts.inference(
                text=text,
                ref_s=speaker_embedding
            )

            if wav is None or len(wav) == 0:
                raise Exception('TTS generation returned empty audio.')

            return age, sex, wav
        
        except Exception as e:
            raise Exception(f'Generation Failed: {e}')